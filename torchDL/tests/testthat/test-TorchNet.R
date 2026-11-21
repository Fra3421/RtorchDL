# Tests for TorchNet: construction, building, training, prediction, export/import.

# ---------------------------------------------------------------------------
# initialize / init_model
# ---------------------------------------------------------------------------

test_that("TorchNet can be instantiated", {
  net <- TorchNet$new()
  expect_s3_class(net, "TorchNet")
  expect_s3_class(net, "R6")
})

test_that("init_model returns the model untouched when FUN is NULL", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- small_model(net)
  expect_identical(net$init_model(m), m)
})

test_that("init_model replaces the model when FUN returns an nn_module", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- small_model(net)
  other <- small_model(net, seed = 2)
  res <- net$init_model(m, function(model) other)
  expect_identical(res, other)
})

test_that("init_model keeps in-place modifications and ignores non-module return values", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- small_model(net)

  zero_all <- function(model) {
    torch::with_no_grad(for (p in model$parameters) p$zero_())
    NULL                                   # not a module: must be ignored
  }
  res <- net$init_model(m, zero_all)

  expect_identical(res, m)
  expect_equal(weight_norm(m), 0)

  m2 <- small_model(net)
  expect_identical(net$init_model(m2, function(model) 42), m2)
})

# ---------------------------------------------------------------------------
# buildFFNN
# ---------------------------------------------------------------------------

test_that("buildFFNN creates the requested architecture", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- net$buildFFNN(input_dim = 3, hidden = c(8, 4), output_dim = 2)

  # 3 linear layers -> 3 weights + 3 biases
  expect_length(m$parameters, 6)

  out <- m(torch::torch_randn(5, 3))
  expect_equal(as.integer(out$shape), c(5L, 2L))
})

test_that("buildFFNN with no hidden layers builds a linear model", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- net$buildFFNN(input_dim = 4, hidden = integer(0), output_dim = 1)
  expect_length(m$parameters, 2)
  expect_equal(as.integer(m(torch::torch_randn(7, 4))$shape), c(7L, 1L))
})

test_that("buildFFNN uses the requested activation (sigmoid by default)", {
  skip_if_no_torch()
  net <- TorchNet$new()

  acts <- c(sigmoid = "nn_sigmoid", tanh = "nn_tanh", silu = "nn_silu",
            softplus = "nn_softplus", gelu = "nn_gelu", relu = "nn_relu")
  for (a in names(acts)) {
    m <- net$buildFFNN(3, 4, activation = a)
    expect_true(inherits(m[[2]], acts[[a]]), info = a)
  }

  m_default <- net$buildFFNN(3, 4)
  expect_true(inherits(m_default[[2]], "nn_sigmoid"))
})

test_that("buildFFNN validates its arguments", {
  skip_if_no_torch()
  net <- TorchNet$new()
  expect_error(net$buildFFNN(3, 4, output_dim = 0), "output_dim")
  expect_error(net$buildFFNN(3, 4, output_dim = 1.5), "output_dim")
  expect_error(net$buildFFNN(3, 4, output_dim = NULL), "output_dim")
  expect_error(net$buildFFNN(3, 4, activation = "not_an_activation"))
})

# ---------------------------------------------------------------------------
# trainNet: argument validation
# ---------------------------------------------------------------------------

test_that("trainNet validates its arguments", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(20)
  m <- small_model(net)
  train <- function(...) net$trainNet(d$x, d$y, model = m, ...)

  expect_error(train(lr = 0), "'lr'")
  expect_error(train(lr = -1), "'lr'")
  expect_error(train(lr = NULL), "lr")
  expect_error(train(epochs = 0), "'epochs'")
  expect_error(train(epochs = 2.5), "'epochs'")
  expect_error(train(epoch_step = 0), "'epoch_step'")
  expect_error(train(epoch_step = 1.5), "'epoch_step'")
  expect_error(train(weight_decay = -1), "'weight_decay'")
  expect_error(train(batch_size = 0), "'batch_size'")
  expect_error(train(batch_size = 2.5), "'batch_size'")
  expect_error(train(loss_name = "not_a_loss"))
  expect_error(train(tau = 1.5), "'tau'")
  expect_error(train(tau = -0.1), "'tau'")
})

test_that("trainNet restores the number of torch threads, also on error", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(20)
  m <- small_model(net)
  old <- torch::torch_get_num_threads()

  net$trainNet(d$x, d$y, model = m, epochs = 1)
  expect_equal(torch::torch_get_num_threads(), old)

  expect_error(net$trainNet(d$x, d$y, model = m, lr = -1))
  expect_equal(torch::torch_get_num_threads(), old)
})

# ---------------------------------------------------------------------------
# trainNet: learning behavior
# ---------------------------------------------------------------------------

test_that("trainNet reduces the error and returns the same (trained) model", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(100)
  m <- small_model(net)

  before <- mean((pred_vec(net, m, d$x) - d$y)^2)
  init <- clone_state(m)

  trained <- net$trainNet(d$x, d$y, model = m, epochs = 300)
  was_training <- trained$training                   # read before predictNet() switches to eval mode
  after <- mean((pred_vec(net, trained, d$x) - d$y)^2)

  expect_identical(trained, m)                       # same object, trained in place
  expect_true(was_training)                          # left in training mode
  expect_false(same_weights(trained, init))          # weights actually changed
  expect_lt(after, before)
  expect_lt(after, 0.5 * var(d$y))                   # clearly better than the mean
})

test_that("trainNet works with mini-batches, including a smaller last batch", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(100)
  m <- small_model(net)
  before <- mean((pred_vec(net, m, d$x) - d$y)^2)

  # 100 observations with batch_size 32 -> batches of 32, 32, 32 and 4
  net$trainNet(d$x, d$y, model = m, epochs = 150, batch_size = 32)
  after <- mean((pred_vec(net, m, d$x) - d$y)^2)
  expect_lt(after, before)
})

test_that("trainNet caps batch_size at the number of observations", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(30)
  m <- small_model(net)
  expect_no_error(net$trainNet(d$x, d$y, model = m, epochs = 2, batch_size = 1000))
})

test_that("trainNet with shuffle = TRUE runs and is reproducible given the seed", {
  skip_if_no_torch()
  fit_once <- function(seed) {
    net <- TorchNet$new()
    d <- make_data(60)
    m <- small_model(net, seed = seed)
    torch::torch_manual_seed(seed)                   # controls the shuffling
    net$trainNet(d$x, d$y, model = m, epochs = 5, batch_size = 16, shuffle = TRUE)
  }

  m1 <- fit_once(7)
  m2 <- fit_once(7)
  m3 <- fit_once(8)

  expect_true(same_weights(m1, m2))
  expect_false(same_weights(m1, m3))
})

test_that("trainNet with shuffle = FALSE is deterministic given the initial weights", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(60)

  m1 <- small_model(net, seed = 1)
  m2 <- small_model(net, seed = 99)
  m2$load_state_dict(clone_state(m1))                # same starting point

  net$trainNet(d$x, d$y, model = m1, epochs = 10, batch_size = 16, shuffle = FALSE)
  net$trainNet(d$x, d$y, model = m2, epochs = 10, batch_size = 16, shuffle = FALSE)
  expect_true(same_weights(m1, m2))
})

test_that("trainNet accepts matrices, data frames and torch tensors", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(40)

  m1 <- small_model(net)
  expect_no_error(net$trainNet(as.data.frame(d$x), d$y, model = m1, epochs = 2))

  m2 <- small_model(net)
  x_t <- torch::torch_tensor(d$x, dtype = torch::torch_float())
  y_t <- torch::torch_tensor(matrix(d$y), dtype = torch::torch_float())
  expect_no_error(net$trainNet(x_t, y_t, model = m2, epochs = 2))
})

# ---------------------------------------------------------------------------
# trainNet: updt hook and regularization
# ---------------------------------------------------------------------------

test_that("updt is called once per mini-batch with (loss, out)", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(50)
  m <- small_model(net)

  calls <- 0
  first_loss_class <- NULL
  first_out_cols <- NULL
  updt <- function(loss, out) {
    calls <<- calls + 1
    if (calls == 1) {
      first_loss_class <<- class(loss)
      first_out_cols <<- out$size(2)
    }
    loss
  }

  # ceiling(50 / 20) = 3 batches per epoch, 3 epochs
  net$trainNet(d$x, d$y, model = m, updt = updt, epochs = 3, batch_size = 20)
  expect_equal(calls, 9)
  expect_true("torch_tensor" %in% first_loss_class)
  expect_equal(first_out_cols, 1)
})

test_that("a penalty added through updt shrinks the weights", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(80)

  base <- small_model(net, seed = 3)
  pen  <- small_model(net, seed = 4)
  pen$load_state_dict(clone_state(base))             # same starting point

  penalty <- function(loss, out) {
    loss + 10 * Reduce(`+`, lapply(pen$parameters, function(p) p$pow(2)$sum()))
  }

  net$trainNet(d$x, d$y, model = base, epochs = 100)
  net$trainNet(d$x, d$y, model = pen, updt = penalty, epochs = 100)
  expect_lt(weight_norm(pen), weight_norm(base))
})

test_that("weight_decay shrinks the weights", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(80)

  free <- small_model(net, seed = 5)
  decayed <- small_model(net, seed = 6)
  decayed$load_state_dict(clone_state(free))

  net$trainNet(d$x, d$y, model = free, epochs = 100, weight_decay = 0)
  net$trainNet(d$x, d$y, model = decayed, epochs = 100, weight_decay = 1)
  expect_lt(weight_norm(decayed), weight_norm(free))
})

# ---------------------------------------------------------------------------
# trainNet: losses and tau
# ---------------------------------------------------------------------------

test_that("every loss family trains without errors and gives finite predictions", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(40)
  losses <- c("huber", "mse", "mae", "logcosh", "expectile", "smooth_pinball")

  for (ln in losses) {
    # symmetric
    m <- small_model(net)
    net$trainNet(d$x, d$y, model = m, loss_name = ln, epochs = 3)
    expect_true(all(is.finite(pred_vec(net, m, d$x))), info = paste(ln, "tau = NULL"))

    # scalar tau
    m <- small_model(net)
    net$trainNet(d$x, d$y, model = m, loss_name = ln, tau = 0.7, epochs = 3)
    expect_true(all(is.finite(pred_vec(net, m, d$x))), info = paste(ln, "tau = 0.7"))

    # vector tau, one output unit per level
    m <- small_model(net, output_dim = 3)
    net$trainNet(d$x, d$y, model = m, loss_name = ln, tau = c(0.1, 0.5, 0.9), epochs = 3)
    out <- torch::as_array(net$predictNet(m, d$x))
    expect_true(all(is.finite(out)), info = paste(ln, "tau vector"))
    expect_equal(dim(out), c(40L, 3L), info = ln)
  }
})

test_that("a vector tau must match the number of output units", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(30)
  taus <- c(0.1, 0.5, 0.9)

  m_bad <- small_model(net, output_dim = 1)
  expect_error(
    net$trainNet(d$x, d$y, model = m_bad, tau = taus, epochs = 2),
    "length\\(tau\\)"
  )

  m_big <- small_model(net, output_dim = 5)
  expect_error(
    net$trainNet(d$x, d$y, model = m_big, tau = taus, epochs = 2),
    "length\\(tau\\)"
  )

  m_ok <- small_model(net, output_dim = 3)
  expect_no_error(net$trainNet(d$x, d$y, model = m_ok, tau = taus, epochs = 2))

  # a scalar tau is not subject to the check
  m_scalar <- small_model(net, output_dim = 1)
  expect_no_error(net$trainNet(d$x, d$y, model = m_scalar, tau = 0.5, epochs = 2))
})

test_that("pinball training estimates the requested conditional quantiles", {
  skip_if_no_torch()
  net <- TorchNet$new()

  set.seed(10)
  n <- 400
  x <- matrix(rnorm(n), ncol = 1)
  y <- x[, 1] + rnorm(n)

  torch::torch_manual_seed(10)
  m <- net$buildFFNN(1, 8, output_dim = 2, activation = "tanh")
  m <- net$trainNet(x, y, model = m, tau = c(0.1, 0.9), loss_name = "mae",
                    epochs = 600, lr = 0.01)

  q <- torch::as_array(net$predictNet(m, x))
  expect_lt(abs(mean(y <= q[, 1]) - 0.1), 0.08)      # empirical coverage ~ 10%
  expect_lt(abs(mean(y <= q[, 2]) - 0.9), 0.08)      # empirical coverage ~ 90%
  expect_gt(mean(q[, 2] - q[, 1]), 0)                # upper quantile above lower one
})

test_that("expectile training recovers the sample expectile (and the mean for tau = NULL)", {
  skip_if_no_torch()
  net <- TorchNet$new()

  set.seed(11)
  n <- 300
  y <- rexp(n)
  x <- matrix(1, nrow = n, ncol = 1)                 # constant input: intercept-only model

  sample_expectile <- function(y, tau) {
    uniroot(function(m) mean(ifelse(y >= m, tau, 1 - tau) * (y - m)),
            range(y))$root
  }

  torch::torch_manual_seed(11)
  m90 <- net$buildFFNN(1, integer(0), output_dim = 1)
  m90 <- net$trainNet(x, y, model = m90, tau = 0.9, loss_name = "expectile",
                      epochs = 800, lr = 0.05)
  expect_lt(abs(mean(pred_vec(net, m90, x)) - sample_expectile(y, 0.9)), 0.1)

  torch::torch_manual_seed(12)
  m_mean <- net$buildFFNN(1, integer(0), output_dim = 1)
  m_mean <- net$trainNet(x, y, model = m_mean, loss_name = "mse",
                         epochs = 800, lr = 0.05)
  expect_lt(abs(mean(pred_vec(net, m_mean, x)) - mean(y)), 0.1)
})

# ---------------------------------------------------------------------------
# trainNet: callbacks, validation and monitoring
# ---------------------------------------------------------------------------

test_that("the callback is reset once and called every epoch_step epochs", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(40)
  m <- small_model(net)
  rec <- RecorderCallback$new()

  net$trainNet(d$x, d$y, model = m, callback = rec, epochs = 20, epoch_step = 5)
  expect_equal(rec$n_reset, 1)
  expect_length(rec$monitors, 4)                     # epochs 5, 10, 15, 20
  expect_equal(rec$steps, rep(5, 4))
})

test_that("training stops as soon as the callback returns FALSE", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(40)
  m <- small_model(net)

  epochs_run <- 0
  updt <- function(loss, out) { epochs_run <<- epochs_run + 1; loss }   # full batch: 1 call/epoch
  rec <- RecorderCallback$new(stop_at = 2)

  net$trainNet(d$x, d$y, model = m, callback = rec, updt = updt,
               epochs = 100, epoch_step = 5)
  expect_length(rec$monitors, 2)
  expect_equal(epochs_run, 10)
})

test_that("the callback is never called when epoch_step exceeds epochs", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(40)
  m <- small_model(net)
  rec <- RecorderCallback$new()

  net$trainNet(d$x, d$y, model = m, callback = rec, epochs = 4, epoch_step = 10)
  expect_length(rec$monitors, 0)
  expect_equal(rec$n_reset, 1)
})

test_that("the monitored value is the validation loss when a validation set is given", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(80)
  idx <- 1:60
  m <- small_model(net)
  rec <- RecorderCallback$new()

  net$trainNet(d$x[idx, ], d$y[idx], d$x[-idx, ], d$y[-idx], model = m,
               callback = rec, epochs = 10, epoch_step = 5, loss_name = "huber")

  yv <- torch::torch_tensor(matrix(d$y[-idx]), dtype = torch::torch_float())
  manual <- TorchLoss$new()$set_loss(NULL, "huber")(net$predictNet(m, d$x[-idx, ]), yv)$item()

  # last check happens at the final epoch, so it matches the final model
  expect_equal(rec$monitors[length(rec$monitors)], manual, tolerance = 1e-4)
})

test_that("without validation data the monitored value is the average training loss", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(50)
  m <- small_model(net)

  y_t <- torch::torch_tensor(matrix(d$y), dtype = torch::torch_float())
  manual <- TorchLoss$new()$set_loss(NULL, "huber")(net$predictNet(m, d$x), y_t)$item()

  # A negligible learning rate keeps the weights fixed, so the weighted mean of the
  # mini-batch losses (batches of 16, 16, 16 and 2) must equal the full-data loss.
  rec <- RecorderCallback$new()
  net$trainNet(d$x, d$y, model = m, callback = rec, epochs = 4, epoch_step = 2,
               batch_size = 16, shuffle = TRUE, lr = 1e-10)
  expect_equal(rec$monitors, rep(manual, 2), tolerance = 1e-4)
})

test_that("a validation set given only in part is ignored", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(50)
  m <- small_model(net)

  y_t <- torch::torch_tensor(matrix(d$y), dtype = torch::torch_float())
  manual <- TorchLoss$new()$set_loss(NULL, "huber")(net$predictNet(m, d$x), y_t)$item()

  rec <- RecorderCallback$new()
  net$trainNet(d$x, d$y, x_val = d$x, y_val = NULL, model = m, callback = rec,
               epochs = 2, epoch_step = 2, lr = 1e-10)
  expect_equal(rec$monitors, manual, tolerance = 1e-4)
})

test_that("the best state returned by the callback is loaded into the model", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(60)

  m <- small_model(net)
  init <- clone_state(m)
  cb <- StateCallback$new(init)
  net$trainNet(d$x, d$y, model = m, callback = cb, epochs = 30, lr = 0.1)
  expect_true(same_weights(m, init))                 # weights rolled back to the stored state

  # sanity check: without such a callback the same training does move the weights
  m2 <- small_model(net)
  net$trainNet(d$x, d$y, model = m2, epochs = 30, lr = 0.1)
  expect_false(same_weights(m2, init))
})

test_that("EarlyStopping stops training after `patience` epochs without improvement", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(40)
  m <- small_model(net)

  epochs_run <- 0
  updt <- function(loss, out) { epochs_run <<- epochs_run + 1; loss }

  # A negligible learning rate means no improvement beyond the first check:
  #   epoch 5: improvement; epoch 10: wait = 5; epoch 15: wait = 10 -> stop.
  es <- EarlyStopping$new(patience = 10)
  net$trainNet(d$x, d$y, model = m, callback = es, updt = updt,
               epochs = 100, epoch_step = 5, lr = 1e-10)
  expect_equal(epochs_run, 15)
})

test_that("EarlyStopping restores the best weights into the model", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(80)
  idx <- 1:60
  m <- small_model(net)
  es <- EarlyStopping$new(patience = 100)

  net$trainNet(d$x[idx, ], d$y[idx], d$x[-idx, ], d$y[-idx], model = m,
               callback = es, epochs = 50, epoch_step = 5)

  expect_false(is.null(es$get_best_state()))
  expect_true(same_weights(m, es$get_best_state()))
})

test_that("an EarlyStopping instance can be reused across trainings", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(60)
  es <- EarlyStopping$new(patience = 100)

  a <- small_model(net, seed = 1)
  net$trainNet(d$x, d$y, model = a, callback = es, epochs = 20, epoch_step = 5)
  a_final <- clone_state(a)

  # With stale state from the first run, the second model would be overwritten with a's weights.
  b <- small_model(net, seed = 2)
  net$trainNet(d$x, d$y, model = b, callback = es, epochs = 20, epoch_step = 5)

  expect_false(same_weights(b, a_final))
  expect_true(same_weights(b, es$get_best_state()))
})

# ---------------------------------------------------------------------------
# predictNet
# ---------------------------------------------------------------------------

test_that("predictNet returns a tensor with one row per observation and no gradient", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(25)
  m <- small_model(net, output_dim = 2)

  p <- net$predictNet(m, d$x)
  expect_s3_class(p, "torch_tensor")
  expect_equal(as.integer(p$shape), c(25L, 2L))
  expect_false(p$requires_grad)
})

test_that("predictNet switches the model to evaluation mode", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(20)
  m <- small_model(net)
  net$trainNet(d$x, d$y, model = m, epochs = 1)
  expect_true(m$training)

  net$predictNet(m, d$x)
  expect_false(m$training)
})

test_that("predictNet accepts matrices, data frames, vectors and tensors consistently", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(15)
  m <- small_model(net)

  p_mat <- pred_vec(net, m, d$x)
  p_df  <- pred_vec(net, m, as.data.frame(d$x))
  p_ten <- pred_vec(net, m, torch::torch_tensor(d$x, dtype = torch::torch_float()))
  expect_equal(p_df, p_mat)
  expect_equal(p_ten, p_mat)

  # a plain vector is treated as a single-feature column
  m1 <- small_model(net, input_dim = 1)
  v <- c(0.1, -0.4, 2)
  expect_equal(pred_vec(net, m1, v), pred_vec(net, m1, matrix(v, ncol = 1)))
})

test_that("predictNet is deterministic", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(15)
  m <- small_model(net)
  expect_identical(pred_vec(net, m, d$x), pred_vec(net, m, d$x))
})

# ---------------------------------------------------------------------------
# exportNet / importNet
# ---------------------------------------------------------------------------

cfg <- list(input_dim = 3, hidden = 8, output_dim = 1, activation = "tanh")

test_that("exportNet / importNet round-trip the model, config and extra objects", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(40)
  m <- net$trainNet(d$x, d$y, model = small_model(net), epochs = 20)

  path <- file.path(withr::local_tempdir(), "nested", "model")
  extra <- list(center = c(1, 2, 3), scale = 2)

  out <- withVisible(net$exportNet(m, path, config = cfg, extra = extra))
  expect_false(out$visible)
  expect_true(dir.exists(out$value))
  expect_equal(basename(out$value), "model")

  expect_true(file.exists(file.path(path, "weights.pt")))
  expect_true(file.exists(file.path(path, "meta.rds")))
  expect_length(list.files(dirname(path), pattern = "\\.tmp_"), 0)   # no temp leftovers

  res <- net$importNet(path)
  expect_named(res, c("model", "extra"))
  expect_false(identical(res$model, m))                       # a new module is built
  expect_false(res$model$training)                            # evaluation mode
  expect_equal(res$extra, extra)
  expect_equal(pred_vec(net, res$model, d$x), pred_vec(net, m, d$x))
})

test_that("exportNet does not modify the model being exported", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- small_model(net)
  before <- clone_state(m)

  net$exportNet(m, file.path(withr::local_tempdir(), "model"), config = cfg)
  expect_true(same_weights(m, before))
})

test_that("exportNet replaces an existing model directory", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(20)
  path <- file.path(withr::local_tempdir(), "model")

  a <- small_model(net, seed = 1)
  b <- small_model(net, seed = 2)
  net$exportNet(a, path, config = cfg)
  net$exportNet(b, path, config = cfg)

  res <- net$importNet(path)
  expect_equal(pred_vec(net, res$model, d$x), pred_vec(net, b, d$x))
  expect_length(list.files(dirname(path), pattern = "\\.tmp_"), 0)
})

test_that("exportNet validates `path`", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- small_model(net)
  expect_error(net$exportNet(m, c("a", "b")), "single directory path")
  expect_error(net$exportNet(m, 1), "single directory path")
})

test_that("importNet needs a model when the export has no config", {
  skip_if_no_torch()
  net <- TorchNet$new()
  d <- make_data(20)
  m <- small_model(net)
  path <- file.path(withr::local_tempdir(), "model")
  net$exportNet(m, path)                                      # no config

  expect_error(net$importNet(path), "without 'config'")

  fresh <- small_model(net, seed = 123)
  res <- net$importNet(path, model = fresh)
  expect_equal(pred_vec(net, res$model, d$x), pred_vec(net, m, d$x))
  expect_null(res$extra)
})

test_that("importNet rejects missing directories and mismatching architectures", {
  skip_if_no_torch()
  net <- TorchNet$new()
  m <- small_model(net)
  path <- file.path(withr::local_tempdir(), "model")
  net$exportNet(m, path, config = cfg)

  expect_error(net$importNet(file.path(withr::local_tempdir(), "missing")), "No valid saved model")

  deeper <- net$buildFFNN(3, c(8, 4), 1, "tanh")              # different number of layers
  expect_error(net$importNet(path, model = deeper), "do not match")
})
