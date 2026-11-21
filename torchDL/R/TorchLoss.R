#' @title TorchLoss Class for Regression, Quantile, and Expectile Loss Functions
#'
#' @description
#' [R6::R6Class] helper that builds \pkg{torch} loss functions for standard
#' regression (Huber, MSE, MAE, log-cosh), quantile regression (pinball and
#' smooth pinball) and expectile / asymmetric Huber regression. The class holds
#' no state: its only public entry point, `$set_loss()`, returns a plain
#' function `function(pred, target)` that computes a scalar loss tensor and can
#' be differentiated with `$backward()`.
#'
#' @details
#' **Choosing the loss.** `$set_loss(tau, type)` selects the loss family through
#' `type` and the symmetric/asymmetric variant through `tau`:
#' \itemize{
#'   \item When `tau` is `NULL`, symmetric (standard) forms are returned. For
#'     `"expectile"` and `"smooth_pinball"`, `tau = 0.5` is used internally.
#'   \item When `tau` is non-`NULL` (a scalar or a numeric vector in `[0, 1]`),
#'     asymmetric or quantile-weighted variants are returned. With a vector
#'     `tau` of length `k`, column `j` of the predictions is evaluated at level
#'     `tau[j]`, so the model must have `k` output units.
#' }
#'
#' The mapping between `type` and the underlying loss is:
#'
#' \tabular{lll}{
#'   `type` \tab `tau = NULL` \tab `tau` supplied \cr
#'   `"huber"` \tab Huber loss (`delta = 1.345`) \tab Asymmetric Huber \cr
#'   `"mse"` \tab Mean squared error \tab Expectile loss (**not** a quantile loss) \cr
#'   `"mae"` \tab Mean absolute error \tab Pinball (quantile) loss \cr
#'   `"logcosh"` \tab Log-cosh loss \tab Asymmetric log-cosh \cr
#'   `"expectile"` \tab Expectile loss at `tau = 0.5` \tab Expectile loss \cr
#'   `"smooth_pinball"` \tab Smooth pinball at `tau = 0.5` \tab Smooth pinball
#' }
#'
#' **Normalization.** All families are scaled so that `tau = 0.5` gives exactly
#' the same value as the corresponding `tau = NULL` version (the asymmetric
#' weights are `2 * tau` and `2 * (1 - tau)`, which equal 1 at `tau = 0.5`).
#' Losses are therefore comparable across the symmetric and asymmetric forms.
#'
#' **Scale of the target.** The Huber threshold (`delta = 1.345`) is expressed
#' in the units of the residuals, so Huber-type losses work best when the
#' target is roughly standardized.
#'
#' **Reduction.** Every loss is averaged over all elements (observations and
#' output columns), returning a scalar tensor.
#'
#' @format An [R6::R6Class] generator object.
#'
#' @seealso [TorchNet], whose `$trainNet()` method builds its loss through this
#'   class.
#'
#' @examples
#' \dontrun{
#' library(torch)
#' pred   <- torch_tensor(matrix(c(0.2, 1.1, 2.3), ncol = 1))
#' target <- torch_tensor(matrix(c(0.0, 1.0, 3.0), ncol = 1))
#'
#' tl <- TorchLoss$new()
#'
#' # Symmetric Huber loss
#' tl$set_loss(tau = NULL, type = "huber")(pred, target)
#'
#' # 90% quantile (pinball) loss
#' tl$set_loss(tau = 0.9, type = "mae")(pred, target)
#'
#' # Normalization check: tau = 0.5 reproduces the symmetric version
#' tl$set_loss(0.5, "mae")(pred, target)    # equals tl$set_loss(NULL, "mae")(pred, target)
#' }
#'
#' @importFrom R6 R6Class
#' @import torch
#' @export
TorchLoss <- R6Class(
  classname = "TorchLoss",

  private = list(
    # ----------------------------------------------------------------------
    # PRIVATE LOSS IMPLEMENTATIONS
    #
    # The methods below are the building blocks used by the public `set_loss()`.
    # They are never called directly by users. Common conventions:
    #
    #   * `pred` and `target` are torch tensors with the same number of rows
    #     (observations). `pred` has k columns (one per output unit); `target`
    #     has either k columns or a single column that is broadcast.
    #   * The residual is always defined as `e = target - pred`.
    #   * `tau` is a numeric scalar or a length-k vector in [0, 1]. It is turned
    #     into a 1 x k tensor so that it broadcasts across rows and each column
    #     of `pred` gets its own level.
    #   * Asymmetric weights are scaled by 2 (2 * tau and 2 * (1 - tau)) so that
    #     tau = 0.5 reproduces the symmetric loss exactly.
    #   * Each function returns a scalar tensor (mean over all elements) that
    #     keeps the autograd graph, ready for `$backward()`.
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # pinball_loss(pred, target, tau)
    #
    # Quantile ("check") loss, the standard loss for quantile regression:
    #     L(e) = 2 * max(tau * e, (tau - 1) * e)
    # Under-predictions (e > 0) are weighted by 2 * tau and over-predictions
    # (e < 0) by 2 * (1 - tau). At tau = 0.5 it equals the absolute error, so it
    # matches `nnf_l1_loss`. It is piecewise linear and not differentiable at
    # e = 0 (see `smooth_pinball_loss` for a smooth alternative).
    # ----------------------------------------------------------------------
    pinball_loss = function(pred, target, tau) {
      tau <- torch::torch_tensor(matrix(tau, nrow = 1), dtype = torch::torch_float())
      e <- target - pred
      2 * torch::torch_maximum(tau * e, (tau - 1) * e)$mean()
    },

    # ----------------------------------------------------------------------
    # asym_huber_loss(pred, target, tau, delta = 1.345)
    #
    # Asymmetric Huber loss: a Huber loss whose value is reweighted by the sign
    # of the residual. The base Huber term is quadratic for |e| <= delta and
    # linear beyond it, which makes it robust to outliers:
    #     h(e) = 0.5 * e^2                      if |e| <= delta
    #            delta * (|e| - 0.5 * delta)    otherwise
    # and the final loss is w(e) * h(e) with w = 2 * tau for e >= 0 and
    # w = 2 * (1 - tau) for e < 0. At tau = 0.5 it equals `nnf_huber_loss`.
    # `delta` is in residual units, so the target should be roughly
    # standardized.
    # ----------------------------------------------------------------------
    asym_huber_loss = function(pred, target, tau, delta = 1.345) {
      tau <- torch::torch_tensor(matrix(tau, nrow = 1), dtype = torch::torch_float())
      e <- target - pred
      a <- torch::torch_abs(e)
      h <- torch::torch_where(a <= delta, 0.5 * e^2, delta * (a - 0.5 * delta))
      w <- torch::torch_where(e >= 0, 2 * tau, 2 * (1 - tau))
      (w * h)$mean()
    },

    # ----------------------------------------------------------------------
    # expectile_loss(pred, target, tau)
    #
    # Asymmetric least-squares (expectile) loss:
    #     L(e) = w(e) * e^2,  with w = 2 * tau for e >= 0 and 2 * (1 - tau)
    #     for e < 0.
    # Estimates the tau-expectile of the conditional distribution (tau = 0.5 is
    # the mean). Unlike the pinball loss it is smooth and differentiable
    # everywhere, but it is NOT a quantile loss: expectiles and quantiles are
    # different summaries of a distribution. At tau = 0.5 it equals the squared
    # error, so it matches `nnf_mse_loss`.
    # ----------------------------------------------------------------------
    expectile_loss = function(pred, target, tau) {
      tau <- torch::torch_tensor(matrix(tau, nrow = 1), dtype = torch::torch_float())
      e <- target - pred
      w <- torch::torch_where(e >= 0, 2 * tau, 2 * (1 - tau))
      (w * e^2)$mean()
    },

    # ----------------------------------------------------------------------
    # logcosh_loss(pred, target, tau = NULL)
    #
    # Log-cosh loss, L(e) = log(cosh(e)): behaves like 0.5 * e^2 for small
    # residuals and like |e| - log(2) for large ones (Huber-like), while being
    # twice continuously differentiable. It is computed in the numerically
    # stable form
    #     log(cosh(e)) = |e| + log1p(exp(-2 * |e|)) - log(2)
    # which avoids the overflow of cosh() for large residuals.
    # If `tau` is supplied, the loss is multiplied by 2 * tau when e >= 0 and by
    # 2 * (1 - tau) when e < 0, giving an asymmetric version (identical to the
    # symmetric one at tau = 0.5). Unlike the other methods, `tau` is optional.
    # ----------------------------------------------------------------------
    logcosh_loss = function(pred, target, tau = NULL) {
      e <- target - pred
      l <- torch::torch_abs(e) + torch::torch_log1p(torch::torch_exp(-2 * torch::torch_abs(e))) - log(2)
      if (!is.null(tau)) {
        tau <- torch::torch_tensor(matrix(tau, nrow = 1), dtype = torch::torch_float())
        l <- torch::torch_where(e >= 0, 2 * tau * l, 2 * (1 - tau) * l)
      }
      l$mean()
    },

    # ----------------------------------------------------------------------
    # smooth_pinball_loss(pred, target, tau, eps = 0.05)
    #
    # Smooth approximation of the pinball loss. The pinball loss can be written
    # as tau * e + max(-e, 0); replacing max(-e, 0) with the softplus function
    #     eps * log(1 + exp(-e / eps))
    # gives a loss that is differentiable everywhere and therefore provides
    # useful gradients near e = 0. The smoothing parameter `eps` controls the
    # approximation: smaller values are closer to the exact pinball loss but
    # less smooth. The result is multiplied by 2 so that tau = 0.5 approximates
    # the absolute error, consistently with `pinball_loss`.
    # ----------------------------------------------------------------------
    smooth_pinball_loss = function(pred, target, tau, eps = 0.05) {
      tau <- torch::torch_tensor(matrix(tau, nrow = 1), dtype = torch::torch_float())
      e <- target - pred
      2 * (tau * e + eps * torch::nnf_softplus(-e / eps))$mean()
    }
  ),

  public = list(
    #' @description Create a `TorchLoss` instance. The class holds no state, so
    #'   nothing needs to be configured.
    #' @return Invisibly returns `self`.
    initialize = function() invisible(self),

    #' @description Build and return a \pkg{torch} loss function
    #'   `function(pred, target)` for the requested loss family and quantile /
    #'   expectile level(s). The returned function computes the mean loss over
    #'   all elements and returns a scalar `torch_tensor` that supports
    #'   backpropagation.
    #' @param tau Numeric scalar, numeric vector with values in `[0, 1]`, or
    #'   `NULL`. This argument has no default and must be supplied explicitly:
    #'   use `NULL` for the symmetric (standard) loss, a scalar for a single
    #'   quantile/expectile level, or a vector of length `k` for `k` levels
    #'   (the model then needs `k` output units, one per level).
    #' @param type Loss family: `"huber"`, `"mse"`, `"mae"`, `"logcosh"`,
    #'   `"expectile"`, or `"smooth_pinball"`. The first value (`"huber"`) is
    #'   used if omitted. Note that `"mse"` with a non-`NULL` `tau` gives the
    #'   expectile loss, whereas the quantile (pinball) loss is obtained with
    #'   `"mae"` or `"smooth_pinball"`.
    #' @return A function `function(p, t)` taking predictions `p` and targets
    #'   `t` (both `torch_tensor`) and returning a scalar loss tensor. Errors if
    #'   `tau` contains values outside `[0, 1]`.
    set_loss = function(tau, type = c("huber", "mse", "mae", "logcosh", "expectile", "smooth_pinball")) {
      type <- match.arg(type)
      if (!is.null(tau) && (any(tau < 0) || any(tau > 1))) stop("'tau' must be in [0,1].")

      if (is.null(tau)) {
        switch(type,
               huber    = function(p, t) private$asym_huber_loss(p, t, 0.5),
               mse      = function(p, t) torch::nnf_mse_loss(p, t),
               mae      = function(p, t) torch::nnf_l1_loss(p, t),
               logcosh  = function(p, t) private$logcosh_loss(p, t),
               expectile = function(p, t) private$expectile_loss(p, t, 0.5),
               smooth_pinball = function(p, t) private$smooth_pinball_loss(p, t, 0.5)
        )
      } else {
        switch(type,
               huber    = function(p, t) private$asym_huber_loss(p, t, tau),
               mse      = function(p, t) private$expectile_loss(p, t, tau),
               mae      = function(p, t) private$pinball_loss(p, t, tau),
               logcosh  = function(p, t) private$logcosh_loss(p, t, tau),
               expectile = function(p, t) private$expectile_loss(p, t, tau),
               smooth_pinball = function(p, t) private$smooth_pinball_loss(p, t, tau)
        )
      }
    }
  )
)
