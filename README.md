# ft_linear_regression

### Index
- [Project Structure](#project-structure)
- [Basic Concepts](#basic-concepts)
  - [Linear Function](#linear-function)
  - [Linear Function Formula](#linear-function-formula)
  - [Graphical Explanation](#graphical-explanation)
  - [Gradient Descent Algorithm](#gradient-descent-algorithm)
    - [Formula](#formula)
    - [Steps](#steps)
- [Calculating Predictions and Errors](#calculating-predictions-and-errors)
- [Parameter Update Step](#parameter-update-step)
- [Cost Calculation](#cost-calculation)
- [Convergence Check](#convergence-check)
- [Return Results](#return-results)
- [Summary](#summary)
  - [Step 1: Start with a Guess (Initialization)](#step-1-start-with-a-guess-initialization)
  - [Step 2: Try to Improve the Line (Gradient Descent)](#step-2-try-to-improve-the-line-gradient-descent)
  - [Step 3: Measure How "Wrong" the Line Is (Predictions and Errors)](#step-3-measure-how-wrong-the-line-is-predictions-and-errors)
  - [Step 4: Adjust the Line (Parameter Updates)](#step-4-adjust-the-line-parameter-updates)
  - [Step 5: Check If We’re Done (Convergence)](#step-5-check-if-were-done-convergence)
  - [Final Result](#final-result)
  - [Simplified Analogy](#simplified-analogy)

### Project Structure

```
README.md
linear_regression/
├── __init__.py
├── train.py          # Main training script
├── predict.py        # Prediction script
├── utils.py          # Utility functions, including `read_csv`
poetry.lock
pyproject.toml
tests/
├── __init__.py
├── test_train.py     # Unit tests for training
├── test_utils.py     # Unit tests for utilities
```


## Basic concepts 

### linear funtion

A **linear function** is a simple type of function where the relationship between the input $(x)$ and output $(y)$ is a straight line when you plot it on a graph.

### Linear Function Formula

It follows this formula:

$$
\large \hat{y} = \theta_0 + \theta_1 x
$$

### where:
- $\large \theta_0$ (intercept): Where the line crosses the y-axis (Where the line starts).  
- $\large \theta_1$ (slope): How steep the line is.  
- $\large x$: Input value (Milage) 
- $\large \hat{y}$: Predicted output(Price). 

### Graphical Explanation

For a more visual understanding, you can refer to this [Graphical explanation](https://www.notion.so/jvalenci/Resolve-question-I-don-t-understand-why-in-this-exercise-we-use-two-thetas-instead-of-jus-1-1599d52658e08088b639c429ed2d8311).

---

### Data Normalization

**Normalization** in data science is the process of scaling your data to a standard range (like 0 to 1). This helps linear regression models perform better by ensuring that all features contribute equally to the result.

**Denormalization** is the reverse process, where you convert the scaled data back to its original scale. This makes it easier to interpret the model's predictions in the context of the original data.

_Example:_

- **Normalization:** Changing ages from a range of 0–100 to 0–1.  
- **Denormalization:** Converting predicted normalized ages back to the 0–100 range.


**Normalization Formula**:
$$
x_{\text{norm}} = \frac{x - \mu_x}{\sigma_x}, \quad y_{\text{norm}} = \frac{y - \mu_y}{\sigma_y}
$$

- $\mu_x$ and $\mu_y$: Means of $x$ and $y$
- $\sigma_x$ and $\sigma_y$: Standard deviations of $x$ and $y$

**Regression Line on Normalized Data**:
$$
y_{\text{norm}} = \theta_0 + \theta_1 \cdot x_{\text{norm}}
$$

**Reverting to Original Scale**:

- **Slope (output - steepness)**:
$$
\text{slope} = \theta_1 \cdot \frac{\sigma_y}{\sigma_x}
$$

- **Intercept (input - line start)**:
$$
\text{intercept} = \mu_y - \text{slope} \cdot \mu_x
$$

**Final Regression Line**:
$$
\hat{y} = \text{intercept} + \text{slope} \cdot x
$$

**Complete Formula**:
$$
\hat{y} = (\theta_0 \cdot \sigma_y + \mu_y) + (\theta_1 \cdot \frac{\sigma_y}{\sigma_x}) \cdot (x - \mu_x)
$$

### Further Reading

For more information about these formulas, see:

- [How to Denormalize Data (Math)](https://jvalenci.notion.site/how-to-denormalize-data-math-15f9d52658e080f9affde29f5c523f02)
- [Normalization and Denormalization: How To](https://www.notion.so/jvalenci/Normalization-and-Denormalization-how-to-1599d52658e0807db023f755f38d091c)

---
### Gradient Descent Algorithm

**What is Gradient Descent?**  
Gradient descent is like finding the lowest point on a hill. You start from a random spot on the hill and step downhill bit by bit until you reach the bottom.

**Goal**  
Its goal is to find the lowest point (the minimum) of a “cost function.” This lowest point makes your model’s predictions as accurate as possible.

### Formula  

$$
\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

$$
\theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i
$$

- **$\theta_0, \theta_1$:** Parameters (intercept and slope)
- **$\alpha$:** Learning rate
- **$\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)$:** Average error
- **$\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i$:** Weighted average error


### Steps: 
- **Initialization:** Code initializes parameters and reads data size, just like math requires starting values for $\theta_0, \theta_1$.
- **Iterations (Gradient Descent):** Code loops and updates parameters, mirroring the iterative approach in the math.
- **Prediction & Error:** Code calculates predictions and errors per data point, following the mathematical definitions.
- **Parameter Update:** Code performs the exact update steps defined by gradient descent formulas.
- **Cost Calculation & Convergence:** Code checks the cost function and stops when changes are small, as mathematically intended.
- **Result:** Code returns final parameters, equivalent to the mathematical concept of having found an approximate minimum of $J(\theta_0, \theta_1)$.
---

### Calculating Predictions and Errors

**Purpose**:
- Focuses on calculating the gradient (the slope of the cost function) to update $\theta_0$ and $\theta_1$ during each iteration of gradient descent.

**Mathematical Concept**:
- Prediction for each data point $(x_i, y_i)$:
$$
  \hat{y}_i = \theta_0 + \theta_1 x_i
$$
- Error:
$$
  e_i = \hat{y}_i - y_i
$$

**Code**:
```python
for i in range(m):
    predicted = theta0 + theta1 * x[i]
    error = predicted - y[i]
    tmp_theta0 += error
    tmp_theta1 += error * x[i]
```

**Parallel Explanation**:
- **In code:** `predicted = theta0 + theta1 * x[i]` mirrors $\hat{y}_i = \theta_0 + \theta_1 x_i$.
- **In code:** `error = predicted - y[i]` matches $e_i = \hat{y}_i - y_i$.
- **In code:** Accumulating `tmp_theta0 += error` and `tmp_theta1 += error * x[i]` corresponds to summing up all $e_i$ and $e_i x_i$ across data points:

$$
  \frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \quad\text{and}\quad \frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)x_i
$$

---

### Parameter Update Step
**Purpose**:
- The purpose of these updates is to adjust the parameters $\theta_0$ and $\theta_1$ to minimize the error between the predicted values ($\hat{y}_i$) and the actual values ($y_i$). This is done using gradient descent, which iteratively updates the parameters in the direction that reduces the error.

**Mathematical Concept**:
- Updating $\theta_0$ and $\theta_1$ using gradient descent:
  
$$
  \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

$$
  \theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i
$$

**Code**:
```python
theta0 -= (learning_rate / m) * tmp_theta0
theta1 -= (learning_rate / m) * tmp_theta1
```

---

### Cost Calculation

**Purpose**:
- `compute_cost`: Evaluates how well the current model (with specific theta0 and theta1) fits the data, providing a numerical measure of error (cost).

**Mathematical Concept**:
- The cost (Mean Squared Error):
$$
  J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

**Code**:
- **In code:** `compute_cost` calculates $\frac{1}{2m}\sum ( \hat{y}_i - y_i )^2$.

---

### Convergence Check

- Convergence means the parameters no longer significantly reduce the cost:
- **Goal**: The goal of checking for convergence is to stop the gradient descent when further iterations do not significantly improve accuracy, saving computational resources and time.

**Mathematical Concept**:


$$
|J_{\text{previous}} - J_{\text{current}}| < \text{tolerance}
$$

**Parameters of the Convergence Formula**:

- **J_previous**: The cost from the previous iteration.
- **J_current**: The cost from the current iteration.
- **tolerance**: A small threshold value to determine if the change in cost is negligible.

**Code**:
```python
if abs(previous_cost - cost) < tolerance:
    print(f"Convergence reached at iteration {it}. Final Cost = {cost:.6f}")
    break

previous_cost = cost
```

**Parallel Explanation**:
- **In code:** If the improvement in cost is less than `tolerance`, we stop.
- **In math:** When changes in $J(\theta_0, \theta_1)$ become negligible, it indicates the solution has (approximately) converged.

---

### Return Results

**Mathematical Concept**:
- After convergence, $\theta_0$ and $\theta_1$ represent the model parameters that minimize the cost.

**Code**:
```python
return theta0, theta1
```

**Parallel Explanation**:
- **In code:** We return the final learned parameters.
- **In math:** This is the conclusion of the optimization process, providing the final model.

---

## Summary

### Understanding Linear Regression with Gradient Descent: A Step-by-Step Guide

### Step 1: Start with a Guess (Initialization)

* **Concept:** You start with a simple line, like a horizontal one ($y = 0$).
* **In Code:** Set the slope ($\theta_1$) and the intercept ($\theta_0$) to 0.

```python
theta0 = 0  # Start with intercept = 0
theta1 = 0  # Start with slope = 0
m = len(x)  # Number of data points
```

---

### Step 2: Try to Improve the Line (Gradient Descent)

* **Concept:** Gradient descent tweaks the line little by little to make it fit the data better.
* **In Code:** For each iteration, adjust $\theta_0$ and $\theta_1$ based on how far off the predictions are.

```python
for it in range(max_iterations):
    tmp_theta0 = 0  # Temporary holder for intercept updates
    tmp_theta1 = 0  # Temporary holder for slope updates
```

---

### Step 3: Measure How "Wrong" the Line Is (Predictions and Errors)

* **Concept:** For every point, calculate how far the line’s prediction ($\hat{y}$) is from the actual value ($y$).
    * Prediction: $\hat{y} = \theta_0 + \theta_1 x$
    * Error: $e = \hat{y} - y$
* **In Code:** Loop through the points, calculate predictions, and sum up errors.

```python
for i in range(m):
    predicted = theta0 + theta1 * x[i]  # Prediction
    error = predicted - y[i]           # Difference from actual value
    tmp_theta0 += error                # Update intercept error
    tmp_theta1 += error * x[i]         # Update slope error
```

---

### Step 4: Adjust the Line (Parameter Updates)

* **Concept:** Move the line closer to the best fit by adjusting $\theta_0$ and $\theta_1$. Use a "learning rate" to control how big the steps are.
    * $\theta_0 := \theta_0 - \alpha \times \text{average error}$
    * $\theta_1 := \theta_1 - \alpha \times \text{average error} \times x$
* **In Code:** Update the parameters.

```python
theta0 -= (learning_rate / m) * tmp_theta0  # Adjust intercept
theta1 -= (learning_rate / m) * tmp_theta1  # Adjust slope
```

---

### Step 5: Check If We’re Done (Convergence)

* **Concept:** Stop when the line barely improves anymore. This happens when the difference in cost (how wrong the model is) becomes very small.
    * Cost = $\frac{1}{2m}\sum (\hat{y} - y)^2$ (average squared error).
* **In Code:** Compare the current cost with the previous cost.

```python
if abs(previous_cost - cost) < tolerance:  # Stop if improvement is tiny
    break
```

---

### Final Result

* **Concept:** After enough iterations, the line is as close as it can get to the best fit. Return the final slope and intercept ($\theta_1$ and $\theta_0$).
* **In Code:** Return the final parameters.

```python
return theta0, theta1  # Best-fit line parameters
```

---

### Simplified Analogy

Imagine you're hiking down a mountain to find the lowest point (best fit for the data):

1. You start at a random spot (initial $\theta_0$ and $\theta_1$).
2. You look around to see the steepest direction downhill (gradient).
3. You take small steps (controlled by the learning rate).
4. You stop when the ground becomes almost flat (cost doesn’t change much).
5. Congratulations! You’re at the best fit line.
