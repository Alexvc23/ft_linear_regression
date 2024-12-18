# ft_linear_regression


## Index
- [Project Structure](#project-structure)
- [Goal of the Project](#goal-of-the-project)
- [Simplified Analogy to Understand the Training Model](#simplified-analogy-to-understand-the-training-model)
- [Basic Concepts](#basic-concepts)
  - [Linear Function](#linear-function)
  - [Data Normalization](#data-normalization)
  - [Gradient Descent Algorithm](#gradient-descent-algorithm)
  - [Calculating Predictions and Errors](#calculating-predictions-and-errors)
  - [Parameter Update Step](#parameter-update-step)
  - [Cost Calculation](#cost-calculation)
  - [Convergence Check](#convergence-check)
  - [Final Result](#final-result)
- [Steps](#steps)
- [Project Structure](#project-structure)

## Goal of the Project

The goal of this project is to  train a linear regression model to predict outcomes based on input data, understand the mathematical foundations behind the model, and evaluate its performance.

So we are going to use the Gradient Descent algorith along with the linear regreation function to train the model and find the best prediction based on the input (data points) 

## Simplified Analogy to undestand the training model


Imagine you're hiking down a mountain to find the lowest point (best fit for the data):


1. You start at a random spot over the  mountain (initial $\theta_0$(staring point) and $\theta_1$ (steepness)).
2. You look around to see the steepest direction downhill (gradient).
3. You take small steps (controlled by the learning rate).
4. You stop when the ground becomes almost flat (cost doesn’t change much).
5. Congratulations! You’re at the best fit line.

## Basic concepts 

### linear funtion

A **linear function** is a simple type of function where the relationship between the input $(x)$ and output $(y)$ is a straight line when you plot it on a graph.


It follows this formula:

$$
\large \hat{y} = \theta_0 + \theta_1 x
$$

**where**:
- $\large \theta_0$ (intercept): Where the line crosses the y-axis (Where the line starts).  
- $\large \theta_1$ (slope): How steep the line is.  
- $\large x$: Input value (Milage) 
- $\large \hat{y}$: Predicted output(Price). 

**Graphical Explanation**

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

---
### Gradient Descent Algorithm

**What is Gradient Descent?**  
Gradient descent is like finding the lowest point on a hill. You start from a random spot on the hill and step downhill bit by bit until you reach the bottom.

**Goal**  
Its goal is to find the lowest point (the minimum) of a “cost function.” This lowest point makes your model’s predictions as accurate as possible.

**Formula**:

$$
\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

$$
\theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i
$$

- **$\theta_0, \theta_1$:** Parameters (intercept and slope)
- **$\alpha$:** Learning rate
- **$m$:** Number of data points
- **$\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)$:** Average error
- **$\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i$:** Weighted average error

---

### Calculating Predictions and Errors

**Purpose**:
- Focuses on calculating the gradient (the slope of the cost function) to update $\theta_0$ and $\theta_1$ during each iteration of gradient descent.

**Mathematical Concept**:
- Prediction for each data point $(x_i, y_i)$:
$$
  \hat{y}_i = \theta_0 + \theta_1 x_i
$$
- Error for each data point:
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
Gradient descent adjusts $\theta_0$ and  $\theta_1$  step by step to reduce the error between predicted  $\hat{y}$  and actual  $y$  values. It uses a small learning step based on the data to find the best fit line.

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

### Final Result

* **Concept:** After enough iterations, the line is as close as it can get to the best fit. Return the final slope and intercept ($\theta_1$ and $\theta_0$).
* **In Code:** Return the final parameters.

```python
return theta0, theta1  # Best-fit line parameters
```

### Steps: 
- **load data**
- **Scale data (Normalize the data)**
- **Initialization:** Code initializes parameters and reads data size, just like math requires starting values for $\theta_0, \theta_1$.
- **Iterations (Gradient Descent):** Code loops and updates parameters, mirroring the iterative approach in the math.
- **Prediction & Error:** Code calculates predictions and errors per data point, following the mathematical definitions.
- **Parameter Update:** Code performs the exact update steps defined by gradient descent formulas.
- **Cost Calculation & Convergence:** Code checks the cost function and stops when changes are small, as mathematically intended.
- **Result:** Code returns final parameters, equivalent to the mathematical concept of having found an approximate minimum of $J(\theta_0, \theta_1)$.
---

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
