# torchDL

 Neural networks for **regression, quantile regression and expectile regression** in R, built on [torch](https://torch.mlverse.org/) and [R6](https://r6.r-lib.org/).

`torchDL` wraps the boilerplate of a `torch` training loop into a small set of composable classes: build a network, train it with Adam using a choice of (a)symmetric losses, monitor a validation set, stop early, then predict, save and reload.

## Features

- **Network builder**: fully connected networks with any number of hidden layers and six activation functions (`sigmoid`, `tanh`, `silu`, `softplus`, `gelu`, `relu`).
- **Training loop**: Adam optimizer, full-batch or mini-batch training, optional shuffling, weight decay, and a hook to customize the loss at every step (e.g. add a penalty).
- **Six loss families**, each with a symmetric and an asymmetric (`tau`) version: Huber, MSE, MAE, log-cosh, expectile, smooth pinball.
- **Multi-quantile / multi-expectile training**: pass a vector `tau` and a network with one output unit per level.
- **Callbacks**: validation monitoring and `EarlyStopping` that restores the best weights; write your own by subclassing `Callback`.
- **Safe persistence**: `exportNet()` / `importNet()` save weights, architecture config and any extra R objects (e.g. scaling constants), with an integrity check on write.

## Installation

`torchDL` requires the [`torch`](https://cran.r-project.org/package=torch) package and its libtorch backend:

```r
install.packages("torch")
torch::install_torch()      # downloads libtorch, only needed once
```

Then install `torchDL` from GitHub:

```r
# install.packages("remotes")
remotes::install_github("<your-username>/torchDL")
```

## Quick start

```r
library(torchDL)
library(torch)

torch_manual_seed(1)
set.seed(1)

# Simulated data
x <- matrix(rnorm(500 * 3), ncol = 3)
y <- x[, 1] - 2 * x[, 2] + rnorm(500, sd = 0.3)
train <- 1:400

net <- TorchNet$new()

# 1. Build the network
model <- net$buildFFNN(input_dim = 3, hidden = c(16, 8), activation = "tanh")

# 2. Train with a validation set and early stopping
model <- net$trainNet(
  x_train = x[train, ], y_train = y[train],
  x_val   = x[-train, ], y_val   = y[-train],
  model      = model,
  callback   = EarlyStopping$new(patience = 50),
  loss_name  = "huber",
  epochs     = 500,
  epoch_step = 5
)

# 3. Predict
pred <- as.numeric(net$predictNet(model, x[-train, ]))
```

`trainNet()` trains the model **in place** and also returns it. When an `EarlyStopping` callback is used, the returned model holds the best weights seen during training, not necessarily the last ones.

## Quantile and expectile regression

Pass `tau` to switch to asymmetric losses. With a vector `tau`, the network must have one output unit per level:

```r
# Three conditional quantiles at once (pinball loss)
q_model <- net$buildFFNN(input_dim = 3, hidden = 16, output_dim = 3)
q_model <- net$trainNet(x, y, model = q_model,
                        tau = c(0.1, 0.5, 0.9), loss_name = "mae")

quantiles <- torch::as_array(net$predictNet(q_model, x))   # n x 3 matrix

# 90% expectile
e_model <- net$buildFFNN(input_dim = 3, hidden = 16)
e_model <- net$trainNet(x, y, model = e_model, tau = 0.9, loss_name = "expectile")
```

If `length(tau)` does not match the number of output units, `trainNet()` stops with an informative error before training starts.

### Loss functions

| `loss_name`        | `tau = NULL`                  | `tau` supplied                                   |
|--------------------|-------------------------------|--------------------------------------------------|
| `"huber"`          | Huber loss (`delta = 1.345`)  | Asymmetric Huber                                 |
| `"mse"`            | Mean squared error            | Expectile loss (**not** a quantile loss)         |
| `"mae"`            | Mean absolute error           | Pinball (quantile) loss                          |
| `"logcosh"`        | Log-cosh                      | Asymmetric log-cosh                              |
| `"expectile"`      | Expectile at `tau = 0.5`      | Expectile loss                                   |
| `"smooth_pinball"` | Smooth pinball at `tau = 0.5` | Smooth pinball (differentiable quantile loss)    |

All families are normalized so that `tau = 0.5` returns exactly the same value as the symmetric version. Huber-type losses use a threshold expressed in residual units, so they work best with a roughly standardized target.

## Custom loss terms

`updt` is called at every mini-batch with the loss and the model output, and must return a loss tensor that is still part of the autograd graph:

```r
l2_penalty <- function(loss, out) {
  loss + 1e-3 * Reduce(`+`, lapply(model$parameters, function(p) p$pow(2)$sum()))
}

model <- net$trainNet(x, y, model = model, updt = l2_penalty)
```

## Callbacks

A callback is notified every `epoch_step` epochs with the monitored loss (the validation loss if a validation set is given, otherwise the average training loss). It returns `TRUE` to continue and `FALSE` to stop, and may store a state dictionary that `trainNet()` loads into the model at the end.

```r
LossLogger <- R6::R6Class(
  "LossLogger",
  inherit = Callback,
  public = list(
    call = function(loss, model, optimizer, step = 1) {
      message("loss = ", signif(loss, 4))
      TRUE
    }
  )
)

model <- net$trainNet(x, y, model = model, callback = LossLogger$new())
```

`EarlyStopping$new(patience, delta)` measures `patience` in **epochs**, so with `epoch_step = 5` and `patience = 20` training stops after 4 consecutive checks without an improvement larger than `delta`. A callback instance is reset automatically at the start of every `trainNet()` call, so it can safely be reused (for example in cross-validation loops).

## Saving and loading

```r
cfg <- list(input_dim = 3, hidden = c(16, 8), output_dim = 1, activation = "tanh")

net$exportNet(model, "models/my_model", config = cfg,
              extra = list(center = colMeans(x), scale = apply(x, 2, sd)))

res <- net$importNet("models/my_model")
res$model      # network with the saved weights, in evaluation mode
res$extra      # the extra objects stored at export time
```

The model is written to a temporary directory, re-read and compared with the in-memory weights, and only then moved into place. If the model was exported without `config`, pass an architecture-compatible network through `importNet(path, model = ...)`.

## Overview of the API

| Class           | Purpose                                      | Main methods                                                                 |
|-----------------|----------------------------------------------|------------------------------------------------------------------------------|
| `TorchNet`      | Build, train, predict, save and load networks | `buildFFNN()`, `init_model()`, `trainNet()`, `predictNet()`, `exportNet()`, `importNet()` |
| `TorchLoss`     | Builds the loss function used in training    | `set_loss(tau, type)`                                                        |
| `Callback`      | Base class (no-op) for training callbacks    | `call()`, `get_best_state()`, `reset()`                                      |
| `EarlyStopping` | Early stopping with best-weights restoration | `call()`, `get_best_state()`, `reset()`                                      |

Full documentation of every class and method is available from the R help system (`?TorchNet`, `?TorchLoss`, `?Callback`, `?EarlyStopping`).

## Notes

- Training runs with a single `torch` thread, restored when `trainNet()` returns.
- For reproducible results set the seed with `torch::torch_manual_seed()`: it controls both weight initialization and mini-batch shuffling.
- Models are returned from `trainNet()` in training mode; `predictNet()` switches them to evaluation mode automatically.

## Testing

The test suite uses [testthat](https://testthat.r-lib.org/) and needs the libtorch backend:

```r
devtools::test()
```

## To-Do
- Add classification
- Add non-FFNN
- Python implementation
