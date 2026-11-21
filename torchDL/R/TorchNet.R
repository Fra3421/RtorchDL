#' @title Feedforward Neural Network Component via Torch
#'
#' @description
#' An [R6::R6Class] component that builds, initializes, trains, saves and loads
#' feedforward neural networks using \pkg{torch}, for regression, quantile
#' regression and expectile regression tasks. Training supports mini-batches,
#' several (optionally asymmetric) loss functions, validation monitoring and
#' pluggable callbacks such as [EarlyStopping].
#'
#' @details
#' The object itself holds no state: it is a toolbox of methods, and every
#' method that works on a network receives the \pkg{torch} `model` explicitly.
#' Typical workflow:
#' \enumerate{
#'   \item `$buildFFNN()` creates the network (optionally customized with
#'     `$init_model()`).
#'   \item `$trainNet()` fits it with Adam.
#'   \item `$predictNet()` generates predictions.
#'   \item `$exportNet()` / `$importNet()` save and restore the trained model.
#' }
#'
#' **Training details (`$trainNet()`).**
#' \itemize{
#'   \item Loss functions are selected via `loss_name` (`"huber"`, `"mse"`,
#'     `"mae"`, `"logcosh"`, `"expectile"`, or `"smooth_pinball"`) and built by
#'     [TorchLoss].
#'   \item When `tau` is `NULL`, standard/symmetric forms are used (for
#'     `"expectile"` and `"smooth_pinball"`, `tau = 0.5` is used internally).
#'   \item When `tau` is provided (scalar or numeric vector in `[0, 1]`),
#'     asymmetric/quantile versions are applied across the specified levels. A
#'     vector `tau` requires a model with one output unit per level.
#'   \item Note that `loss_name = "mse"` with a non-`NULL` `tau` computes the
#'     asymmetric expectile loss (`expectile_loss`), not a quantile (pinball)
#'     loss; use `loss_name = "mae"` (or `"smooth_pinball"`) with `tau` for
#'     quantile regression.
#'   \item All loss families are normalized so that `tau = 0.5` gives exactly
#'     the same value as the corresponding `tau = NULL` version.
#'   \item If a validation set (`x_val` and `y_val`) is provided, the validation
#'     loss is evaluated every `epoch_step` epochs and passed to `callback`;
#'     otherwise the average training loss of the epoch is used.
#'   \item At the end of training, if `callback` stores a best state, those
#'     weights are restored into the returned model.
#'   \item The number of \pkg{torch} threads is temporarily set to `1` during
#'     training and restored on exit.
#' }
#'
#' @return An `R6` object of class `"TorchNet"`.
#'
#' @seealso [TorchLoss] for the available losses, [Callback] and
#'   [EarlyStopping] for the callback interface.
#'
#' @examples
#' \dontrun{
#' library(torch)
#' torch_manual_seed(1)
#'
#' # Simulated regression data
#' x <- matrix(rnorm(500 * 3), ncol = 3)
#' y <- x[, 1] - 2 * x[, 2] + rnorm(500, sd = 0.3)
#' idx <- 1:400
#'
#' net   <- TorchNet$new()
#' model <- net$buildFFNN(input_dim = 3, hidden = c(16, 8), activation = "tanh")
#'
#' # Train with a validation set and early stopping
#' model <- net$trainNet(x[idx, ], y[idx], x[-idx, ], y[-idx], model = model,
#'                       callback = EarlyStopping$new(patience = 50),
#'                       loss_name = "huber", epochs = 300)
#' pred <- as.numeric(net$predictNet(model, x[-idx, ]))
#'
#' # Quantile regression: three output units, one per quantile level
#' q_model <- net$buildFFNN(input_dim = 3, hidden = 16, output_dim = 3)
#' q_model <- net$trainNet(x, y, model = q_model, tau = c(0.1, 0.5, 0.9),
#'                         loss_name = "mae")
#'
#' # Save and reload (the config allows rebuilding the architecture)
#' cfg <- list(input_dim = 3, hidden = c(16, 8), output_dim = 1, activation = "tanh")
#' path <- file.path(tempdir(), "my_model")
#' net$exportNet(model, path, config = cfg)
#' res <- net$importNet(path)
#' }
#'
#' @importFrom R6 R6Class
#' @import torch
#' @export
TorchNet <- R6Class(
  classname = "TorchNet",

  private = list(
    # ----------------------------------------------------------------------
    # PRIVATE HELPERS
    #
    # Internal utilities used by the public methods. They are not part of the
    # user-facing API and may change without notice.
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # set_activation(act)
    #
    # Maps the name of an activation function to the corresponding torch module
    # *constructor* (not an instance). `act` must be one of "sigmoid", "tanh",
    # "silu", "softplus", "gelu" or "relu". The caller instantiates it with
    # `act()` once for every hidden layer, so each layer gets its own module.
    # The name is validated upstream with `match.arg()` in `buildFFNN()`; an
    # unknown name would make `switch()` return NULL.
    # ----------------------------------------------------------------------
    set_activation = function(act) {
      make_act <- switch(act,
                         sigmoid  = torch::nn_sigmoid,
                         tanh     = torch::nn_tanh,
                         silu     = torch::nn_silu,
                         softplus = torch::nn_softplus,
                         gelu     = torch::nn_gelu,
                         relu     = torch::nn_relu)

      return(make_act)
    },

    # ----------------------------------------------------------------------
    # to_torch_tensor(x)
    #
    # Normalizes user input into a torch tensor. If `x` is already a
    # `torch_tensor` (e.g. a pre-built tensor, possibly with more than two
    # dimensions) it is returned untouched, so its dtype and shape are
    # preserved. Otherwise `x` (vector, matrix or data frame of numbers) is
    # coerced with `as.matrix()` and converted to a float32 tensor. A plain
    # vector therefore becomes a single-column (n x 1) matrix, which is the
    # shape expected for targets and for one-feature inputs.
    # ----------------------------------------------------------------------
    to_torch_tensor = function(x) {
      if (inherits(x, "torch_tensor")) return(x)          # allow tensors as input
      torch::torch_tensor(as.matrix(x), dtype = torch::torch_float())
    }
  ),

  public = list(
    #' @description Create a `TorchNet` component. The object holds no state, so
    #'   nothing needs to be configured: every method that operates on a
    #'   network receives the `model` explicitly.
    #' @return Invisibly returns `self`.
    initialize = function() invisible(self),

    #' @description Apply an optional custom initialization function to a
    #'   \pkg{torch} model, e.g. to set a particular weight initialization
    #'   scheme. If no function is given, the model is returned unchanged and
    #'   keeps \pkg{torch}'s default initialization.
    #' @param model A \pkg{torch} neural network module.
    #' @param FUN Optional function taking `model`. It may modify the module in
    #'   place (\pkg{torch} modules are reference objects, so changes made inside
    #'   `FUN`, e.g. under `with_no_grad()`, persist) and/or return a module. If
    #'   it returns an `nn_module`, that module replaces `model`; any other
    #'   return value is ignored (default `NULL`).
    #' @return The initialized \pkg{torch} model.
    init_model = function(model, FUN = NULL) {
      if (!is.null(FUN)) {
        res <- FUN(model)
        if (inherits(res, "nn_module")) model <- res
      }
      model
    },

    #' @description Build a sequential feedforward neural network
    #'   (`nn_sequential`): for each hidden layer a fully connected linear layer
    #'   followed by the chosen activation, and a final linear output layer with
    #'   no activation. Weights use \pkg{torch}'s default initialization.
    #' @param input_dim Integer. Number of input features.
    #' @param hidden Integer vector giving the number of units in each hidden
    #'   layer, in order (e.g. `c(32, 16)` builds two hidden layers). An empty
    #'   vector (`integer(0)`) builds a purely linear model.
    #' @param output_dim Integer `>= 1`. Number of output units (default `1`).
    #'   Use one unit per quantile/expectile level when training with a vector
    #'   `tau`.
    #' @param activation Activation function applied after each hidden linear
    #'   layer: `"sigmoid"`, `"tanh"`, `"silu"`, `"softplus"`, `"gelu"`, or
    #'   `"relu"` (default `"sigmoid"`).
    #' @return A \pkg{torch} `nn_sequential` module. Errors if `output_dim` is
    #'   not an integer `>= 1`.
    buildFFNN = function(input_dim, hidden, output_dim = 1, activation = c("sigmoid", "tanh", "silu", "softplus", "gelu", "relu")) {
      if (is.null(output_dim) || output_dim %% 1 != 0 || output_dim < 1) stop("'output_dim' must be an integer >= 1.")
      act <- private$set_activation(match.arg(activation))             # set torch activation function

      layers <- list()
      in_d <- input_dim
      for (h in hidden) {
        layers[[length(layers) + 1]] <- torch::nn_linear(in_d, h)      # linear combinations
        layers[[length(layers) + 1]] <- act()                          # apply non-linearity
        in_d <- h
      }

      layers[[length(layers) + 1]] <- torch::nn_linear(in_d, as.integer(output_dim))        # linear final output
      do.call(torch::nn_sequential, layers)                            # the final model structure
    },

    #' @description Train a \pkg{torch} neural network with the Adam optimizer.
    #'   Each epoch visits all training observations in (optionally shuffled)
    #'   mini-batches. Every `epoch_step` epochs the monitored loss is computed
    #'   and passed to `callback`, which may stop the training; at the end, the
    #'   best weights stored by the callback (if any) are restored into the
    #'   model.
    #'
    #'   The monitored loss is the validation loss when `x_val` and `y_val` are
    #'   both supplied, and otherwise the average training loss of the epoch
    #'   (mini-batch losses weighted by batch size, computed before each
    #'   parameter update and including any effect of `updt`). The validation
    #'   loss uses the plain loss function and does not apply `updt`. If only
    #'   one of `x_val` and `y_val` is supplied, the validation set is ignored.
    #'
    #'   Before training, the model is checked once on a single row: when `tau`
    #'   has more than one value, the number of output units must match
    #'   `length(tau)`. The training is performed with a single \pkg{torch}
    #'   thread, restored on exit. Reproducibility requires setting the seed
    #'   with `torch::torch_manual_seed()` (it governs shuffling and default
    #'   weight initialization).
    #' @param x_train Training predictors: numeric matrix, vector, data frame,
    #'   or `torch_tensor`. Rows are observations.
    #' @param y_train Training targets: numeric vector, matrix, or
    #'   `torch_tensor`, with the same number of rows as `x_train`. A vector is
    #'   treated as a single column.
    #' @param x_val Optional validation predictors (default `NULL`).
    #' @param y_val Optional validation targets (default `NULL`).
    #' @param model A \pkg{torch} neural network module to train. It is
    #'   modified in place and also returned.
    #' @param callback Optional callback object implementing the [Callback]
    #'   interface (`$reset()`, `$call(monitor, model, optimizer, step)` and
    #'   `$get_best_state()`), e.g. an [EarlyStopping] instance. If `NULL`, a
    #'   no-op `Callback$new()` is used. The callback is reset at the start of
    #'   every call, so an instance can be reused (default `NULL`).
    #' @param updt Optional hook `function(loss, out)` called at every
    #'   mini-batch step, after the loss is computed and before
    #'   backpropagation. It receives the loss tensor and the model output for
    #'   the batch and must return a loss tensor still attached to the autograd
    #'   graph (e.g. the loss plus a penalty term). If `NULL`, the loss is used
    #'   unchanged (default `NULL`).
    #' @param tau Optional numeric scalar or vector in `[0, 1]` controlling
    #'   quantile/expectile/asymmetric loss weighting (default `NULL`, i.e.
    #'   symmetric losses). See [TorchLoss].
    #' @param lr Learning rate for the Adam optimizer, `> 0` (default `0.01`).
    #' @param epochs Integer `>= 1`. Maximum number of training epochs
    #'   (default `500`).
    #' @param epoch_step Integer `>= 1`. Interval, in epochs, at which the
    #'   monitored loss is computed and `callback` is evaluated (default `5`).
    #'   If larger than `epochs`, the callback is never evaluated.
    #' @param weight_decay Non-negative L2 weight-decay penalty passed to Adam
    #'   (default `0`).
    #' @param batch_size Optional mini-batch size as a positive integer, capped
    #'   at the number of observations. If `NULL`, full-batch training is used
    #'   (default `NULL`).
    #' @param shuffle Logical. If `TRUE`, observations are randomly reordered at
    #'   every epoch before forming mini-batches (default `TRUE`).
    #' @param loss_name Loss function family: `"huber"`, `"mse"`, `"mae"`,
    #'   `"logcosh"`, `"expectile"`, or `"smooth_pinball"` (default
    #'   `"huber"`).
    #' @return The trained \pkg{torch} model (the same object passed in
    #'   `model`), left in training mode. Use `$predictNet()` for predictions,
    #'   which switches to evaluation mode automatically. Errors on invalid
    #'   arguments or if `length(tau) > 1` does not match the number of output
    #'   units.
    trainNet = function(x_train, y_train, x_val = NULL, y_val = NULL, model, callback = NULL, updt = NULL,
                        tau = NULL, lr = 0.01, epochs = 500, epoch_step = 5, weight_decay = 0,
                        batch_size = NULL, shuffle = TRUE,
                        loss_name = c("huber", "mse", "mae", "logcosh", "expectile", "smooth_pinball")) {

      # process sequentially
      old <- torch::torch_get_num_threads()
      torch::torch_set_num_threads(1)
      on.exit(torch::torch_set_num_threads(old))

      # Control inputs
      if (is.null(lr) || lr <= 0) stop("'lr' must be > 0.")
      if (is.null(epochs) || epochs %% 1 != 0 || epochs < 1) stop("'epochs' must be integer and >= 1.")
      if (is.null(epoch_step) || epoch_step %% 1 != 0 || epoch_step < 1) stop("'epoch_step' must be integer and >= 1.")
      if (is.null(weight_decay) || weight_decay < 0) stop("'weight_decay' must be >= 0.")
      if (!is.null(batch_size) && (batch_size %% 1 != 0 || batch_size < 1)) stop("'batch_size' must be an integer >= 1.")

      # If callback and updt are null, set them in way they don't touch the process
      if (is.null(updt)) updt <- function(loss, out) loss
      if (is.null(callback)) callback <- Callback$new()
      callback$reset()                                       # reset the callback object status

      # Convert training data to torch tensors
      x_tensor <- private$to_torch_tensor(x_train)
      y_tensor <- private$to_torch_tensor(y_train)

      # Mini-batches setup
      n <- x_tensor$size(1)
      batch_size <- if (is.null(batch_size)) n else min(as.integer(batch_size), n)
      n_batches <- ceiling(n / batch_size)

      # Set torch loss function
      loss_name <- match.arg(loss_name)
      loss_fn <- TorchLoss$new()$set_loss(tau, loss_name)

      # if validation set is present create function for computing loss on validation set
      val_loss <- NULL
      if (!is.null(x_val) && !is.null(y_val)) {
        # Convert validation data to torch tensors
        xv <- private$to_torch_tensor(x_val)
        yv <- private$to_torch_tensor(y_val)

        val_loss <- function() {
          model$eval()
          v <- torch::with_no_grad(loss_fn(model(xv), yv))$item()
          model$train()
          v
        }
      }

      # Initialize optimizer
      optimizer <- torch::optim_adam(model$parameters, lr = lr, weight_decay = weight_decay)

      # Check if length of tau is equal to the length of output dimensions
      if (!is.null(tau) && length(tau) > 1) {
        model$eval()
        k <- torch::with_no_grad(model(x_tensor[1:1, ..]))$size(2)   # dry run on a single row
        model$train()
        if (k != length(tau))
          stop("Model output dimension (", k, ") must equal length(tau) (", length(tau), ").")
      }

      # train across epochs
      for (epoch in seq_len(epochs)) {
        model$train()                             # set to 'train' modality
        perm <- if (shuffle) torch::torch_randperm(n) + 1L else torch::torch_arange(1, n, dtype = torch::torch_long())      # Generate sample indexes
        check <- (epoch %% epoch_step == 0)       # check if we are at 'step' epoch
        epoch_loss <- 0                           # total loss of the epoch

        # for each mini-batch
        for (b in seq_len(n_batches)) {
          i  <- perm[((b - 1) * batch_size + 1):min(b * batch_size, n)]     # extract indices corresponding to batch 'b', avoiding going out of bounds
          xb <- x_tensor[i, ..]          # select sampled rows from regressors
          yb <- y_tensor[i, ..]          # select sampled rows from target

          optimizer$zero_grad()          # delete gradients of previous epoch (for default, they sum)
          out  <- model(xb)              # predict
          loss <- loss_fn(out, yb)       # compute loss
          loss <- updt(loss, out)        # update loss parameters with custom function
          loss$backward()                # back-propagation
          optimizer$step()               # compute model weights with back-propagated gradients and lr

          if (check && is.null(val_loss)) epoch_loss <- epoch_loss + loss$item() * i$size(1)    # epoch loss as weighted mean of mini-batches losses (weights = cardinality of the batch)
        }

        if (check) {
          monitor  <- if (is.null(val_loss)) epoch_loss / n else val_loss()              # extract actual loss (if present, from validation)
          call_out <- callback$call(monitor, model, optimizer, step = epoch_step)        # custom callback function (e.g. EarlyStopping)
          if (!isTRUE(call_out)) break
        }
      }

      # If EarlyStopping was applied, set the best weights
      best_state <- callback$get_best_state()
      if (!is.null(best_state)) model$load_state_dict(best_state)

      return(model)
    },

    #' @description Generate predictions from a trained \pkg{torch} model. The
    #'   model is switched to evaluation mode (and left in it) and the forward
    #'   pass runs without gradient tracking.
    #' @param model A \pkg{torch} neural network module.
    #' @param new_x New predictors to generate predictions for: numeric matrix,
    #'   vector, data frame, or `torch_tensor`. Rows are observations and the
    #'   columns must match the model's input dimension.
    #' @return A `torch_tensor` with one row per observation and one column per
    #'   output unit. Use `as.numeric()` or `as.matrix()` to convert it to an R
    #'   object.
    predictNet = function(model, new_x) {
      model$eval()
      torch::with_no_grad({
        preds <- model(private$to_torch_tensor(new_x))
      })

      return(preds)
    },

    #' @description Save a trained \pkg{torch} model to a directory. The weights
    #'   are stored with \pkg{torch}'s native serialization (`torch_save`) in
    #'   `weights.pt`, and a small R file `meta.rds` stores the architecture
    #'   `config`, the `extra` objects and the parameter names.
    #'
    #'   The new files are first written to a temporary directory next to
    #'   `path`, re-read and compared with the in-memory weights to verify
    #'   integrity, and only then moved to `path`, so a failed or corrupted
    #'   write never produces a half-written model; the temporary directory is
    #'   removed on failure. If `path` already exists it is deleted just before
    #'   the final rename, so that last swap is not strictly atomic. Safe to call
    #'   from a parallel worker before it shuts down.
    #' @param model A trained \pkg{torch} neural network module.
    #' @param path Single character string: directory where the model is stored.
    #'   Parent directories are created; an existing directory at `path` is
    #'   replaced.
    #' @param config Optional named list of arguments for `$buildFFNN()` (e.g.
    #'   `list(input_dim = 3, hidden = c(16, 8), output_dim = 1, activation =
    #'   "tanh")`). If given, `$importNet()` can rebuild the architecture
    #'   without being handed a model.
    #' @param extra Optional list of additional R objects to store alongside the
    #'   weights (e.g. scaling attributes of the data).
    #' @return Invisibly returns the normalized `path`. Errors if `path` is not a
    #'   single string, if the integrity check fails, or if the directory cannot
    #'   be moved into place.
    exportNet = function(model, path, config = NULL, extra = NULL) {
      if (!is.character(path) || length(path) != 1) stop("'path' must be a single directory path.")
      path <- normalizePath(path, mustWork = FALSE)

      # unique temp dir in the same parent folder (same filesystem -> rename is atomic)
      dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
      tmp <- paste0(path, ".tmp_", Sys.getpid(), "_", format(Sys.time(), "%H%M%OS3"))
      dir.create(tmp)
      on.exit(unlink(tmp, recursive = TRUE), add = TRUE)      # cleanup if anything fails

      # weights: detached CPU copies of the state_dict, saved natively by torch
      state <- lapply(model$state_dict(), function(t) t$detach()$cpu()$clone())
      torch::torch_save(state, file.path(tmp, "weights.pt"))
      saveRDS(list(config = config, extra = extra, names = names(state)), file.path(tmp, "meta.rds"))

      # integrity check: reload and compare
      chk <- torch::torch_load(file.path(tmp, "weights.pt"))
      ok <- setequal(names(chk), names(state)) &&
        all(vapply(names(state), function(nm) torch::torch_equal(chk[[nm]], state[[nm]]), logical(1)))
      if (!ok) stop("Integrity check failed: saved weights differ from the model weights.")

      # swap into place
      if (dir.exists(path)) unlink(path, recursive = TRUE)
      if (!file.rename(tmp, path)) stop("Could not move the model to '", path, "'.")
      invisible(path)
    },

    #' @description Load a model previously saved with `$exportNet()`. The
    #'   architecture is rebuilt from the stored `config` via `$buildFFNN()`
    #'   unless a `model` is supplied; the saved weights are then loaded into it
    #'   and the model is put in evaluation mode.
    #' @param path Directory created by `$exportNet()` (it must contain
    #'   `weights.pt` and `meta.rds`).
    #' @param model Optional \pkg{torch} module with the same architecture as
    #'   the saved one. Required only if the model was saved without `config`;
    #'   if supplied, it is used instead of rebuilding from `config`, and it is
    #'   modified in place.
    #' @return A list with two elements: `model`, the \pkg{torch} module with
    #'   the loaded weights (in evaluation mode), and `extra`, the additional
    #'   objects stored at export time (`NULL` if none). Errors if the
    #'   directory is not a valid saved model, if neither `config` nor `model`
    #'   is available, or if the saved weights do not match the architecture.
    importNet = function(path, model = NULL) {
      w <- file.path(path, "weights.pt")
      m <- file.path(path, "meta.rds")
      if (!file.exists(w) || !file.exists(m)) stop("No valid saved model found in: ", path)

      meta  <- readRDS(m)
      state <- torch::torch_load(w)

      if (is.null(model)) {
        if (is.null(meta$config)) stop("Model saved without 'config': provide a 'model' with the same architecture.")
        model <- do.call(self$buildFFNN, meta$config)
      }

      if (!setequal(names(model$state_dict()), names(state)))
        stop("Saved weights do not match the model architecture.")

      model$load_state_dict(state)
      model$eval()
      list(model = model, extra = meta$extra)
    }
  )
)
