# Tests for Callback (base class) and EarlyStopping.

# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

test_that("the base Callback is a no-op", {
  cb <- Callback$new()
  expect_s3_class(cb, "Callback")
  expect_true(cb$call(loss = 1, model = NULL, optimizer = NULL, step = 5))
  expect_null(cb$get_best_state())
  expect_null(cb$reset())
})

test_that("subclasses can override the callback interface", {
  rec <- RecorderCallback$new(stop_at = 2)
  expect_s3_class(rec, "Callback")

  expect_true(rec$call(1.0, NULL, NULL, step = 5))
  expect_false(rec$call(0.9, NULL, NULL, step = 5))
  expect_equal(rec$monitors, c(1.0, 0.9))

  rec$reset()
  expect_length(rec$monitors, 0)
})

# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

test_that("EarlyStopping inherits from Callback and validates its arguments", {
  es <- EarlyStopping$new()
  expect_s3_class(es, "EarlyStopping")
  expect_s3_class(es, "Callback")
  expect_null(es$get_best_state())

  expect_error(EarlyStopping$new(patience = 0), "'patience'")
  expect_error(EarlyStopping$new(patience = 1.5), "'patience'")
  expect_error(EarlyStopping$new(patience = NULL), "'patience'")
  expect_error(EarlyStopping$new(delta = 0), "'delta'")
  expect_error(EarlyStopping$new(delta = -1), "'delta'")
  expect_error(EarlyStopping$new(delta = NULL), "'delta'")
})

test_that("EarlyStopping counts patience in epochs, not in checks", {
  skip_if_no_torch()
  model <- torch::nn_linear(2, 1)
  es <- EarlyStopping$new(patience = 10)

  expect_true(es$call(1.00, model, NULL, step = 5))      # improvement, wait = 0
  expect_true(es$call(0.90, model, NULL, step = 5))      # improvement, wait = 0
  expect_true(es$call(0.95, model, NULL, step = 5))      # no improvement, wait = 5
  expect_false(es$call(0.95, model, NULL, step = 5))     # no improvement, wait = 10 -> stop
})

test_that("an improvement resets the waiting counter", {
  skip_if_no_torch()
  model <- torch::nn_linear(2, 1)
  es <- EarlyStopping$new(patience = 10)

  es$call(1.0, model, NULL, step = 5)
  expect_true(es$call(1.0, model, NULL, step = 5))       # wait = 5
  expect_true(es$call(0.5, model, NULL, step = 5))       # improvement: wait back to 0
  expect_true(es$call(0.5, model, NULL, step = 5))       # wait = 5
  expect_false(es$call(0.5, model, NULL, step = 5))      # wait = 10 -> stop
})

test_that("EarlyStopping only counts decreases larger than delta as improvements", {
  skip_if_no_torch()
  model <- torch::nn_linear(2, 1)
  es <- EarlyStopping$new(patience = 100, delta = 0.1)

  torch::with_no_grad(model$weight$fill_(1))
  es$call(1.0, model, NULL, step = 1)
  expect_equal(as.numeric(es$get_best_state()$weight$detach()), c(1, 1))

  torch::with_no_grad(model$weight$fill_(5))
  es$call(0.95, model, NULL, step = 1)                   # decrease of 0.05 < delta: ignored
  expect_equal(as.numeric(es$get_best_state()$weight$detach()), c(1, 1))

  es$call(0.5, model, NULL, step = 1)                    # decrease larger than delta: stored
  expect_equal(as.numeric(es$get_best_state()$weight$detach()), c(5, 5))
})

test_that("the stored best state is a clone, independent of later model updates", {
  skip_if_no_torch()
  model <- torch::nn_linear(2, 1)
  es <- EarlyStopping$new()

  torch::with_no_grad(model$weight$fill_(2))
  es$call(1.0, model, NULL, step = 1)
  torch::with_no_grad(model$weight$fill_(9))             # modify the model afterwards

  expect_equal(as.numeric(es$get_best_state()$weight$detach()), c(2, 2))
})

test_that("non-finite monitored values count as non-improvements but never stop on their own", {
  skip_if_no_torch()
  model <- torch::nn_linear(2, 1)

  # no snapshot is taken for non-finite values, and `wait` grows without stopping
  es <- EarlyStopping$new(patience = 10)
  expect_true(es$call(NaN, model, NULL, step = 5))
  expect_true(es$call(Inf, model, NULL, step = 5))
  expect_true(es$call(NA_real_, model, NULL, step = 5))
  expect_null(es$get_best_state())

  # the accumulated wait does count at the next finite, non-improving check
  es2 <- EarlyStopping$new(patience = 10)
  es2$call(1.0, model, NULL, step = 5)                   # improvement
  es2$call(NaN, model, NULL, step = 5)                   # wait = 5
  es2$call(NaN, model, NULL, step = 5)                   # wait = 10
  expect_false(es2$call(2.0, model, NULL, step = 5))     # wait = 15 -> stop

  # a finite improvement after non-finite values recovers normally
  es3 <- EarlyStopping$new(patience = 10)
  es3$call(NaN, model, NULL, step = 5)
  expect_true(es3$call(1.0, model, NULL, step = 5))
  expect_false(is.null(es3$get_best_state()))
})

test_that("reset clears the state but keeps patience and delta", {
  skip_if_no_torch()
  model <- torch::nn_linear(2, 1)
  es <- EarlyStopping$new(patience = 10)

  es$call(0.1, model, NULL, step = 5)                    # very low best loss
  es$call(0.1, model, NULL, step = 5)                    # wait = 5
  expect_false(is.null(es$get_best_state()))

  es$reset()
  expect_null(es$get_best_state())

  # a much higher loss is an improvement again, because best_loss went back to Inf
  expect_true(es$call(100, model, NULL, step = 5))
  expect_false(is.null(es$get_best_state()))

  # patience is still 10 and the counter restarted: wait = 5, then 10 -> stop
  expect_true(es$call(100, model, NULL, step = 5))
  expect_false(es$call(100, model, NULL, step = 5))
})
