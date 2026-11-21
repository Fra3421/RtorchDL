# Tests for TorchLoss: formulas, tau = 0.5 normalization, asymmetry and gradients.

tt <- function(m) torch::torch_tensor(m, dtype = torch::torch_float())

# Reference implementations in plain R (e = target - pred)
ref_pinball <- function(e, tau) mean(2 * pmax(tau * e, (tau - 1) * e))
ref_expectile <- function(e, tau) mean(ifelse(e >= 0, 2 * tau, 2 * (1 - tau)) * e^2)
ref_huber <- function(e, tau = 0.5, delta = 1.345) {
  h <- ifelse(abs(e) <= delta, 0.5 * e^2, delta * (abs(e) - 0.5 * delta))
  mean(ifelse(e >= 0, 2 * tau, 2 * (1 - tau)) * h)
}
ref_logcosh <- function(e, tau = 0.5) {
  mean(ifelse(e >= 0, 2 * tau, 2 * (1 - tau)) * log(cosh(e)))
}
ref_smooth_pinball <- function(e, tau, eps = 0.05) {
  2 * mean(tau * e + eps * log1p(exp(-e / eps)))
}

pred_v   <- c(0.2, 1.1, 2.3, -0.5, 0.0, 1.0)
target_v <- c(0.0, 1.0, 3.0,  0.4, 0.0, -2.0)
e_v <- target_v - pred_v

test_that("set_loss returns a function and validates tau and type", {
  skip_if_no_torch()
  tl <- TorchLoss$new()
  expect_s3_class(tl, "TorchLoss")
  expect_type(tl$set_loss(NULL, "huber"), "closure")

  expect_error(tl$set_loss(-0.1, "mae"), "'tau'")
  expect_error(tl$set_loss(1.1, "mae"), "'tau'")
  expect_error(tl$set_loss(c(0.5, 2), "mae"), "'tau'")
  expect_error(tl$set_loss(NULL, "not_a_loss"))
})

test_that("symmetric losses match their definitions", {
  skip_if_no_torch()
  tl <- TorchLoss$new()
  p <- tt(matrix(pred_v)); t <- tt(matrix(target_v))

  expect_equal(tl$set_loss(NULL, "mse")(p, t)$item(), mean(e_v^2), tolerance = 1e-4)
  expect_equal(tl$set_loss(NULL, "mae")(p, t)$item(), mean(abs(e_v)), tolerance = 1e-4)
  expect_equal(tl$set_loss(NULL, "huber")(p, t)$item(), ref_huber(e_v), tolerance = 1e-4)
  expect_equal(tl$set_loss(NULL, "logcosh")(p, t)$item(), ref_logcosh(e_v), tolerance = 1e-4)
  expect_equal(tl$set_loss(NULL, "expectile")(p, t)$item(), mean(e_v^2), tolerance = 1e-4)
  expect_equal(tl$set_loss(NULL, "smooth_pinball")(p, t)$item(),
               ref_smooth_pinball(e_v, 0.5), tolerance = 1e-4)
})

test_that("asymmetric losses match their definitions", {
  skip_if_no_torch()
  tl <- TorchLoss$new()
  p <- tt(matrix(pred_v)); t <- tt(matrix(target_v))
  tau <- 0.8

  expect_equal(tl$set_loss(tau, "mae")(p, t)$item(), ref_pinball(e_v, tau), tolerance = 1e-4)
  expect_equal(tl$set_loss(tau, "mse")(p, t)$item(), ref_expectile(e_v, tau), tolerance = 1e-4)
  expect_equal(tl$set_loss(tau, "expectile")(p, t)$item(), ref_expectile(e_v, tau), tolerance = 1e-4)
  expect_equal(tl$set_loss(tau, "huber")(p, t)$item(), ref_huber(e_v, tau), tolerance = 1e-4)
  expect_equal(tl$set_loss(tau, "logcosh")(p, t)$item(), ref_logcosh(e_v, tau), tolerance = 1e-4)
  expect_equal(tl$set_loss(tau, "smooth_pinball")(p, t)$item(),
               ref_smooth_pinball(e_v, tau), tolerance = 1e-4)
})

test_that("tau = 0.5 reproduces the symmetric version of every loss family", {
  skip_if_no_torch()
  tl <- TorchLoss$new()
  set.seed(3)
  p <- tt(matrix(rnorm(50)))
  t <- tt(matrix(rnorm(50)))

  for (type in c("huber", "mse", "mae", "logcosh", "expectile", "smooth_pinball")) {
    sym  <- tl$set_loss(NULL, type)(p, t)$item()
    asym <- tl$set_loss(0.5, type)(p, t)$item()
    expect_equal(asym, sym, tolerance = 1e-4, info = type)
  }
})

test_that("a vector tau applies one level per output column", {
  skip_if_no_torch()
  tl <- TorchLoss$new()
  taus <- c(0.1, 0.5, 0.9)
  n <- length(pred_v)

  pred_mat <- matrix(rep(pred_v, 3) + rep(c(-0.3, 0, 0.3), each = n), ncol = 3)
  e_mat <- target_v - pred_mat                           # target recycled down the columns
  tau_mat <- matrix(taus, nrow = n, ncol = 3, byrow = TRUE)

  p <- tt(pred_mat); t <- tt(matrix(target_v))
  w <- ifelse(e_mat >= 0, 2 * tau_mat, 2 * (1 - tau_mat))

  expect_equal(tl$set_loss(taus, "mae")(p, t)$item(),
               mean(2 * pmax(tau_mat * e_mat, (tau_mat - 1) * e_mat)), tolerance = 1e-4)
  expect_equal(tl$set_loss(taus, "expectile")(p, t)$item(), mean(w * e_mat^2), tolerance = 1e-4)
})

test_that("pinball loss penalizes under-prediction more when tau is high", {
  skip_if_no_torch()
  loss <- TorchLoss$new()$set_loss(0.9, "mae")
  under <- loss(tt(matrix(0)), tt(matrix(1)))$item()     # prediction below the target
  over  <- loss(tt(matrix(1)), tt(matrix(0)))$item()     # prediction above the target
  expect_equal(under, 1.8, tolerance = 1e-5)             # 2 * 0.9 * 1
  expect_equal(over, 0.2, tolerance = 1e-5)              # 2 * 0.1 * 1
})

test_that("log-cosh stays finite and linear for very large residuals", {
  skip_if_no_torch()
  loss <- TorchLoss$new()$set_loss(NULL, "logcosh")
  val <- loss(tt(matrix(0)), tt(matrix(1000)))$item()
  expect_true(is.finite(val))
  expect_equal(val, 1000 - log(2), tolerance = 1e-4)
})

test_that("smooth pinball is close to the exact pinball loss", {
  skip_if_no_torch()
  tl <- TorchLoss$new()
  set.seed(4)
  p <- tt(matrix(rnorm(200)))
  t <- tt(matrix(rnorm(200)))
  for (tau in c(0.1, 0.5, 0.9)) {
    exact  <- tl$set_loss(tau, "mae")(p, t)$item()
    smooth <- tl$set_loss(tau, "smooth_pinball")(p, t)$item()
    expect_lt(abs(smooth - exact), 0.1, label = paste("tau =", tau))   # bound: 2 * eps * log(2)
  }
})

test_that("all losses are differentiable with finite gradients", {
  skip_if_no_torch()
  tl <- TorchLoss$new()
  for (type in c("huber", "mse", "mae", "logcosh", "expectile", "smooth_pinball")) {
    for (tau in list(NULL, 0.3)) {
      p <- torch::torch_tensor(matrix(pred_v), dtype = torch::torch_float(), requires_grad = TRUE)
      loss <- tl$set_loss(tau, type)(p, tt(matrix(target_v)))
      loss$backward()
      g <- as.numeric(torch::as_array(p$grad))
      expect_true(all(is.finite(g)), info = paste(type, "tau =", format(tau)))
      expect_gt(sum(abs(g)), 0, label = paste(type, "tau =", format(tau)))
    }
  }
})
