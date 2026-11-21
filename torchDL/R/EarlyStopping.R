#' @title EarlyStopping Callback for Model Training
#'
#' @description
#' [R6::R6Class] callback inheriting from [Callback] that monitors a loss across
#' training checks, keeps a cloned copy of the best model weights seen so far,
#' and tells the training loop to stop once the loss has not improved for
#' `patience` epochs. At the end of `TorchNet$trainNet()` the best weights are
#' restored into the model, so the returned model is the best one observed and
#' not necessarily the last one.
#'
#' @details
#' **What counts as an improvement.** At each check, the monitored value
#' improves if it is lower than `best_loss - delta`, i.e. it must beat the best
#' value by an absolute margin of at least `delta`. On improvement the callback
#' stores a snapshot of the weights and resets its waiting counter to zero.
#'
#' **Patience is measured in epochs, not in checks.** `trainNet()` evaluates the
#' callback every `epoch_step` epochs and passes that value as `step`. Each
#' check without improvement adds `step` to the waiting counter, and training
#' stops as soon as the counter reaches `patience`. For example, with
#' `epoch_step = 5` and `patience = 20`, training stops after 4 consecutive
#' checks without improvement (20 epochs).
#'
#' **Non-finite values.** If the monitored loss is `NaN`, `Inf` or `NA`, the
#' check counts as a non-improvement (the waiting counter still grows) but does
#' not stop training by itself, and no snapshot is taken. Training stops at the
#' first subsequent finite, non-improving check once the counter has reached
#' `patience`.
#'
#' **Reuse.** Call `$reset()` (done automatically by `trainNet()`) to clear the
#' best loss, the stored weights and the counter before starting a new training.
#'
#' @format An [R6::R6Class] generator object inheriting from [Callback].
#'
#' @seealso [Callback] for the callback interface, and [TorchNet] for the
#'   training loop.
#'
#' @examples
#' \dontrun{
#' net   <- TorchNet$new()
#' model <- net$buildFFNN(input_dim = 3, hidden = c(16, 8))
#'
#' # Stop after 50 epochs without a validation-loss improvement of at least 1e-4,
#' # checking every 5 epochs (i.e. after 10 consecutive checks without progress)
#' es <- EarlyStopping$new(patience = 50, delta = 1e-4)
#' model <- net$trainNet(x_train, y_train, x_val, y_val, model = model,
#'                       callback = es, epochs = 1000, epoch_step = 5)
#' }
#'
#' @importFrom R6 R6Class
#' @import torch
#' @export
EarlyStopping <- R6Class(
  classname = "EarlyStopping",
  inherit = Callback,

  private = list(
    # ----------------------------------------------------------------------
    # Internal state of the callback (all fields are cleared by `$reset()`).
    #
    #   wait       Number of epochs elapsed since the last improvement. It grows
    #              by `step` at every check and is set back to 0 whenever the
    #              monitored loss improves.
    #   best_loss  Lowest monitored loss recorded so far (`Inf` before the first
    #              finite check). A new value must be lower than
    #              `best_loss - delta` to be considered an improvement.
    #   best_state Cloned state dictionary (named list of tensors) of the model
    #              at the epoch that produced `best_loss`; `NULL` until the
    #              first improvement.
    #   patience   Maximum number of epochs allowed without improvement before
    #              training is stopped. Set by `$initialize()`.
    #   delta      Minimum absolute decrease in the monitored loss required to
    #              count as an improvement. Set by `$initialize()`.
    # ----------------------------------------------------------------------
    wait = 0, best_loss = Inf, best_state = NULL, patience = 20, delta = 1e-5
  ),

  public = list(
    #' @description Create an `EarlyStopping` callback.
    #' @param patience Positive integer. Number of **epochs** without sufficient
    #'   improvement tolerated before training is stopped (default `20`). Since
    #'   the callback is evaluated every `epoch_step` epochs, this corresponds to
    #'   `patience / epoch_step` consecutive checks.
    #' @param delta Positive numeric. Minimum absolute decrease of the monitored
    #'   loss (`best_loss - delta`) required to count as an improvement
    #'   (default `1e-5`).
    #' @return Not used directly: `$new()` returns the new `EarlyStopping`
    #'   object. Errors if `patience` is not an integer `>= 1` or `delta` is not
    #'   `> 0`.
    initialize = function(patience = 20, delta = 1e-5) {
      if (is.null(patience) || patience %% 1 != 0 || patience < 1) stop("'patience' must be integer and >= 1.")
      if (is.null(delta) || delta <= 0) stop("'delta' must be > 0.")
      private$patience <- patience
      private$delta <- delta
    },

    #' @description Evaluate the early-stopping rule at a check point. The
    #'   waiting counter is increased by `step`; if `monitor` improves on the
    #'   best loss by more than `delta`, the model weights are cloned and stored
    #'   and the counter is reset; otherwise the callback decides whether
    #'   `patience` has been exhausted. Non-finite values of `monitor` are
    #'   treated as non-improvements that never stop training on their own.
    #' @param monitor Numeric scalar. Monitored loss at the current check point
    #'   (validation loss if a validation set was given to `trainNet()`,
    #'   otherwise the average training loss of the epoch). Corresponds to the
    #'   `loss` argument of [Callback]`$call()`.
    #' @param model A \pkg{torch} neural network module providing
    #'   `$state_dict()`; its weights are cloned when `monitor` improves.
    #' @param optimizer The optimizer passed by the training loop (unused).
    #' @param step Integer. Number of epochs since the previous check, added to
    #'   the waiting counter at each call (default `5`; `trainNet()` always
    #'   passes its own `epoch_step`).
    #' @return `TRUE` if training should continue (improvement found, or fewer
    #'   than `patience` epochs elapsed without improvement), `FALSE` if early
    #'   stopping is triggered.
    call = function(monitor, model, optimizer, step = 5) {
      private$wait <- private$wait + step
      if (!is.finite(monitor)) return(TRUE)

      if (monitor < private$best_loss - private$delta) {
        private$best_loss  <- monitor
        private$best_state <- lapply(model$state_dict(), function(t) t$clone())
        private$wait <- 0
        return(TRUE)
      }

      return(private$wait < private$patience)
    },

    #' @description Retrieve the weights of the best epoch observed so far.
    #'   `trainNet()` loads them into the model at the end of training.
    #' @return Named list of cloned \pkg{torch} tensors (a state dictionary)
    #'   corresponding to the best monitored loss, or `NULL` if no improvement
    #'   has been recorded yet (e.g. training ended before the first check).
    get_best_state = function() private$best_state,

    #' @description Clear the internal state (waiting counter, best loss and
    #'   stored weights) so that the same callback instance can be reused for a
    #'   new training. `patience` and `delta` are kept. Called automatically at
    #'   the beginning of `TorchNet$trainNet()`.
    #' @return `NULL`, invisibly. The return value is ignored by `trainNet()`.
    reset = function() {
      private$wait <- 0
      private$best_loss <- Inf
      private$best_state <- NULL
      invisible(NULL)
    }
  )
)
