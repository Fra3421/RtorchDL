# Shared helpers for the test suite (sourced automatically by testthat).

# Skip a test when torch (or its libtorch backend) is not available.
skip_if_no_torch <- function() {
  testthat::skip_if_not_installed("torch")
  testthat::skip_if_not(torch::torch_is_installed(), "libtorch backend is not installed")
}

# Simulated regression data: y = 2 * x1 - x2 + noise.
make_data <- function(n = 100, p = 3, noise = 0.1, seed = 1) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), ncol = p)
  y <- 2 * x[, 1] - x[, 2] + rnorm(n, sd = noise)
  list(x = x, y = y)
}

# Small FFNN built with a fixed torch seed (reproducible initial weights).
small_model <- function(net, input_dim = 3, hidden = 8, output_dim = 1,
                        activation = "tanh", seed = 1) {
  torch::torch_manual_seed(seed)
  net$buildFFNN(input_dim, hidden, output_dim, activation)
}

# Detached clone of a model's state dictionary.
clone_state <- function(model) {
  lapply(model$state_dict(), function(t) t$detach()$clone())
}

# TRUE if two models / state dictionaries hold (numerically) the same weights.
same_weights <- function(a, b, tol = 1e-6) {
  sa <- if (inherits(a, "nn_module")) a$state_dict() else a
  sb <- if (inherits(b, "nn_module")) b$state_dict() else b
  if (!setequal(names(sa), names(sb))) return(FALSE)
  all(vapply(names(sa), function(nm) {
    torch::torch_allclose(sa[[nm]], sb[[nm]], rtol = tol, atol = tol)
  }, logical(1)))
}

# Sum of squared parameters (squared L2 norm of all weights and biases).
weight_norm <- function(model) {
  sum(vapply(model$parameters, function(p) p$pow(2)$sum()$item(), numeric(1)))
}

# Predictions as a plain numeric vector (single-output models).
pred_vec <- function(net, model, x) {
  as.numeric(torch::as_array(net$predictNet(model, x)))
}

# Callback that records every call and can stop training after `stop_at` calls.
RecorderCallback <- R6::R6Class(
  "RecorderCallback",
  inherit = Callback,
  public = list(
    monitors = numeric(0), steps = numeric(0), n_reset = 0, stop_at = Inf,
    initialize = function(stop_at = Inf) self$stop_at <- stop_at,
    reset = function() {
      self$n_reset <- self$n_reset + 1
      self$monitors <- numeric(0)
      self$steps <- numeric(0)
      NULL
    },
    call = function(loss, model, optimizer, step = 1) {
      self$monitors <- c(self$monitors, loss)
      self$steps <- c(self$steps, step)
      length(self$monitors) < self$stop_at
    }
  )
)

# Callback that hands a fixed state dictionary back to the training loop.
StateCallback <- R6::R6Class(
  "StateCallback",
  inherit = Callback,
  public = list(
    state = NULL,
    initialize = function(state) self$state <- state,
    get_best_state = function() self$state
  )
)
