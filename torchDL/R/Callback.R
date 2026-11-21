#' @title Callback Base Class for Model Training Loops
#'
#' @description
#' Base [R6::R6Class] defining the (no-op) callback interface used by
#' `TorchNet$trainNet()`. A callback is notified every `epoch_step` epochs with
#' the monitored loss, may store a snapshot of the model, and may tell the
#' training loop to stop. The base class does nothing: it never stops training
#' and never stores any state. Subclasses (such as [EarlyStopping]) override the
#' public methods to implement actual behavior.
#'
#' @details
#' **Callback contract.** `trainNet()` interacts with a callback object through
#' exactly three methods, in this order:
#' \enumerate{
#'   \item `$reset()` is called once, before training starts, so that a callback
#'     instance can safely be reused across several trainings (e.g. in
#'     cross-validation or hyper-parameter tuning loops).
#'   \item `$call(monitor, model, optimizer, step)` is called every `epoch_step`
#'     epochs. It must return `TRUE` to continue training and anything else
#'     (typically `FALSE`) to stop. `trainNet()` calls it with positional
#'     arguments for the first three, and `step` by name.
#'   \item `$get_best_state()` is called once, after training ends (normally or
#'     through an early stop). If it returns a non-`NULL` state dictionary,
#'     `trainNet()` loads it into the model before returning it.
#' }
#'
#' To write a custom callback, inherit from `Callback` and override whichever of
#' these methods you need; keep the same signatures.
#'
#' @format An [R6::R6Class] generator object.
#'
#' @seealso [EarlyStopping] for a ready-to-use subclass, and [TorchNet] for the
#'   training loop that consumes callbacks.
#'
#' @examples
#' # The base callback never stops training and stores nothing
#' cb <- Callback$new()
#' cb$call(loss = 0.5, model = NULL, optimizer = NULL, step = 5)  # TRUE
#' cb$get_best_state()                                            # NULL
#'
#' # A custom callback that logs the monitored loss and stops after `max_checks` checks
#' Logger <- R6::R6Class(
#'   "Logger",
#'   inherit = Callback,
#'   public = list(
#'     max_checks = 3, n = 0,
#'     call = function(loss, model, optimizer, step = 1) {
#'       self$n <- self$n + 1
#'       message("check ", self$n, ": loss = ", signif(loss, 4))
#'       self$n < self$max_checks
#'     },
#'     reset = function() { self$n <- 0; invisible(NULL) }
#'   )
#' )
#'
#' @importFrom R6 R6Class
#' @export
Callback <- R6Class(
  classname = "Callback",

  public = list(
    #' @description Create a `Callback` instance. The base class holds no state,
    #'   so nothing needs to be configured.
    #' @return Not used directly: `$new()` returns the new `Callback` object.
    initialize = function() {},

    #' @description Hook executed by the training loop at every check point
    #'   (every `epoch_step` epochs). Use it to monitor the loss, snapshot the
    #'   model, or request an early stop.
    #' @param loss Numeric scalar. The monitored loss at the current check
    #'   point: the validation loss if a validation set was supplied to
    #'   `trainNet()`, otherwise the average training loss over the epoch.
    #' @param model A \pkg{torch} neural network module being trained. It can
    #'   be inspected (e.g. `model$state_dict()`) but should not be modified.
    #' @param optimizer The \pkg{torch} optimizer driving parameter updates
    #'   (e.g. to read or change the learning rate).
    #' @param step Integer. Number of epochs elapsed since the previous call,
    #'   i.e. the `epoch_step` used by `trainNet()` (default `1`).
    #' @return `TRUE` to continue training, any other value (typically `FALSE`)
    #'   to stop it. The base implementation always returns `TRUE`.
    call = function(loss, model, optimizer, step = 1) TRUE,

    #' @description Retrieve the best model state tracked by the callback, if
    #'   any. When a non-`NULL` value is returned, `trainNet()` loads it into the
    #'   model at the end of training.
    #' @return A named list of \pkg{torch} tensors (a state dictionary, as
    #'   returned by `model$state_dict()`), or `NULL` if no state is stored.
    #'   The base implementation always returns `NULL`.
    get_best_state = function() NULL,

    #' @description Reset the internal state of the callback to its initial
    #'   values. `trainNet()` calls it at the start of every training, so that a
    #'   single instance can be reused without carrying over information (best
    #'   loss, saved weights, counters) from a previous run. The base
    #'   implementation has no state and does nothing.
    #' @return `NULL`. The return value is ignored by `trainNet()`.
    reset = function() NULL
  )
)
