# ML From Scratch

A learning repository for building small machine learning models in Python and understanding the math behind their predictions and training.

Starting with a C++ and JavaScript background, I'm working through prediction functions, loss, gradients, and evaluation before moving to larger models. Each exercise keeps the calculations visible rather than hiding them behind a machine learning framework.

## Current exercises

| Exercise | Status | Concepts |
| --- | --- | --- |
| [Linear regression](fundamentals/linear_regression.py) | Implemented | One input, mean squared error, gradient descent, separate training and test examples |
| [Multiple linear regression](fundamentals/multiple_linear_regression.py) | Starter dataset only | Next: two inputs, two weights, one bias |

The current exercises use plain Python and synthetic data. No third-party packages or GPU are required.

## Run an exercise

From the repository root, create a virtual environment if you don't already have one:

```bash
python -m venv .venv
```

On Windows using Git Bash:

```bash
source .venv/Scripts/activate
python fundamentals/linear_regression.py
python fundamentals/multiple_linear_regression.py
```

On Windows using PowerShell, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe fundamentals\linear_regression.py
.\.venv\Scripts\python.exe fundamentals\multiple_linear_regression.py
```

On macOS or Linux, activate with `source .venv/bin/activate`, then run the same `python fundamentals/...` commands.

## What the first model does

The prediction rule is:

```text
prediction = weight * x + bias
```

Training starts weight and bias at zero, then makes 1,000 passes through three examples: `(1, 6)`, `(2, 11)`, and `(3, 13)`. Each pass computes errors and updates both parameters using the gradients of mean squared error.

The learned values are approximately:

```text
weight = 3.5
bias = 3.0
training loss = 0.5
```

The points do not lie exactly on a straight line, so the best-fitting line still has nonzero loss.

After training, the script evaluates separate examples without updating the parameters:

| Input | Target | Prediction |
| --- | --- | --- |
| 4 | 18 | 17.0 |
| 5 | 22 | 20.5 |

The test mean squared error is approximately **1.625**. These tiny, invented datasets demonstrate the evaluation process; they do not establish real-world predictive performance. The test inputs also lie outside the training input range.

Every execution trains from zero. Parameters stay in memory during that run and are not saved to disk.

## Next steps

- Implement multiple-input prediction and understand each weight's role.
- Extend the loss and gradient calculations to multiple inputs.
- Visualize model fit and training progress.
- Practice evaluation on larger datasets and compare against simple baselines.

A possible separate portfolio project will investigate whether information extracted from news or company statements adds predictive value beyond numerical market data. This repository is for the fundamentals leading up to that work; no market-data pipeline or trading system is implemented here.

## Earlier scaffolding

The repository history includes an earlier MNIST direction, and `scripts/get_mnist.py` is a data-download utility from that setup. It is not needed for the current exercises. Local environments and downloaded data are excluded from Git through `.gitignore`.
