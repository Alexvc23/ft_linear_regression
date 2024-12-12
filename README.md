# ft_linear_regression

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

## Business logic

Below is a parallel, step-by-step comparison of the code execution and the corresponding mathematical concepts. This will help you see how each line of code relates directly to the underlying theory.

---

### Initialization Step

**Mathematical Concept**:  
- **Parameters**: $\theta_0$ and $\theta_1$ represent the intercept and slope of the linear model.
- **Data Size**: $m = \text{number of data points}$.

**Code**:
```python
theta0 = 0         # Mathematically: θ₀ = 0
theta1 = 0         # Mathematically: θ₁ = 0
m = len(x)          # m = number of data points
previous_cost = float('inf')  # Keep track of cost from previous iteration
```

**Parallel Explanation**:
- **In code:** We set $\theta_0$ and $\theta_1$ to zero, meaning our initial guess is a horizontal line.
- **In math:** Before learning begins, parameters are usually initialized to some starting value (often 0).
- **In code:** We determine $m$, the size of the dataset, just as the theory suggests we need to know how many points we’re averaging over.

---

### Gradient Descent Iteration

**Mathematical Concept**:
- We will run multiple iterations of gradient descent:
$$
  \text{Repeat until convergence or max\_iterations:}
$$

**Code**:
```python
for it in range(max_iterations):
    tmp_theta0 = 0
    tmp_theta1 = 0
```

**Parallel Explanation**:
- **In code:** The loop represents the iterative process of gradient descent.
- **In math:** Gradient descent is an iterative optimization algorithm. Each iteration attempts to move $\theta_0$ and $\theta_1$ closer to values that minimize the cost.
- **In code:** `tmp_theta0` and `tmp_theta1` are initialized to accumulate gradients.  
- **In math:** These accumulators represent the summations $\sum_{i=1}^{m} (\hat{y}_i - y_i)$ and $\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i$.

---

### Calculating Predictions and Errors

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

  \sum_{i=1}^{m} (\hat{y}_i - y_i) \quad\text{and}\quad \sum_{i=1}^{m} (\hat{y}_i - y_i)x_i

$$
---

### Parameter Update Step

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

**Parallel Explanation**:
- **In code:** `learning_rate` is $\alpha$.
- **In math:** Dividing by $m$ gives the average gradient.
- **In code:** `theta0 -= ...` and `theta1 -= ...` literally implement the formula for updating the parameters based on the averaged gradients.
- This step moves $\theta_0$ and $\theta_1$ in the direction that reduces the cost.

---

### Cost Calculation

**Mathematical Concept**:
- The cost (Mean Squared Error):
  \[
  J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2
  \]

**Code**:
```python
cost = compute_cost(theta0, theta1, x, y, m)
```

**Parallel Explanation**:
- **In code:** `compute_cost` calculates $\frac{1}{2m}\sum ( \hat{y}_i - y_i )^2$.
- **In math:** $J(\theta_0, \theta_1)$ is the measure we aim to minimize with gradient descent.

---

### Convergence Check

**Mathematical Concept**:
- Convergence means the parameters no longer significantly reduce the cost:
$$
  |J_{\text{previous}} - J_{\text{current}}| < \text{tolerance}
$$

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

### Summary

- **Initialization:** Code initializes parameters and reads data size, just like math requires starting values for $\theta_0, \theta_1$.
- **Iterations (Gradient Descent):** Code loops and updates parameters, mirroring the iterative approach in the math.
- **Prediction & Error:** Code calculates predictions and errors per data point, following the mathematical definitions.
- **Parameter Update:** Code performs the exact update steps defined by gradient descent formulas.
- **Cost Calculation & Convergence:** Code checks the cost function and stops when changes are small, as mathematically intended.
- **Result:** Code returns final parameters, equivalent to the mathematical concept of having found an approximate minimum of $J(\theta_0, \theta_1)$.

In essence, each code block corresponds to a mathematical step: from initializing parameters, computing predictions and errors, summing gradients, updating parameters according to the gradient descent formula, checking for convergence, and finally returning the optimal parameters.