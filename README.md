
# ft_linear_regression
- [English version](#english-version)
- [French version ](#french-version)

## English version

## Index
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
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone the repository](#step-1-clone-the-repository)
  - [Step 2: Configure the environment](#step-2-configure-the-environment)
  - [Step 3: Enter Poetry Shell](#step-3-enter-poetry-shell)
  - [Step 4: Train the Model](#step-4-train-the-model)
  - [Step 5: Verify the Setup](#step-5-verify-the-setup)

## Goal of the Project

The goal of this project is to  train a linear regression model to predict outcomes based on input data, understand the mathematical foundations behind the model, and evaluate its performance.

So we are going to use the Gradient Descent algorith along with the linear regreation function to train the model and find the best prediction based on the input (data points) 

## Simplified Analogy to Understand the Training Model

Imagine you're hiking down a mountain to find the lowest point (best fit for the data):

1. You start at a random spot on the mountain (initial $\theta_0$ (starting point) and $\theta_1$ (steepness)).
2. You look around to see the steepest direction downhill (gradient).
3. You take small steps (controlled by the learning rate).
4. You adjust your direction at each step to keep descending (parameter update).
5. You regularly check how close you are to the bottom (cost calculation).
6. You stop when the ground becomes almost flat (cost doesn’t change much).
7. Congratulations! You’ve found the best fit line.

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

- **Calculate the <span style="color:lightblue">**prediction**</span></bold> for all the data points using the input data points
- **In code:** `predicted = theta0 + theta1 * x[i]`
- **mirrors**:

$$
\hat{y}_i = \theta_0 + \theta_1 x_i
$$

- Calcule the <span style="color:red">**error**</span> for all the data points using the output data points
  - **In code:** `error = predicted - y[i]` 
  - **Matches**:


$$
\text{Error}=\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

- **In code:** Accumulating `tmp_theta0 += error` and `tmp_theta1 += error * x[i]` corresponds to calculating all the errors for $\theta_0$ and $\theta_1$ 

$$
 \text{Erreur}\theta_0 =Error ----> \text{Erreur}\theta_0=\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

$$
\text{Erreur}\theta_1 = (Error)x_i---->\;\;\text{Erreur}\theta_1 =\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i
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
\frac{|J_{\text{previous}} - J_{\text{current}}|}{J_{\text{previous}}} < \text{tolerance}
$$

**Parameters of the Convergence Formula**:

- **J_previous**: The cost from the previous iteration.
- **J_current**: The cost from the current iteration.
- **tolerance**: A small threshold value to determine if the change in cost is negligible.

**Code**:
```python
current_tolerance = abs((previous_cost - cost) / previous_cost)  # Relative tolerance
print(f"Iteration {it}: Cost = {cost:.6f}, Relative Tolerance = {current_tolerance:.6f}")
if current_tolerance < tolerance:  # Check if the relative improvement is below the threshold
  print(f"Convergence reached at iteration {it}. Final Cost = {cost:.6f}")
  break

previous_cost = cost
```

**Parallel Explanation**:
- **In code:** If the relative improvement in cost is less than `tolerance`, we stop.
- **In math:** When changes in $J(\theta_0, \theta_1)$ become negligible relative to their current value, it indicates the solution has (approximately) converged.

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
├── README.md
├── assets
│   └── data.csv
├── linear_regression
│   ├── __init__.py
│   ├── predict.py
│   ├── tools.py
│   └── train.py
├── poetry.lock
├── pyproject.toml
├── tests
│   ├── __init__.py
│   └── test_tools.py
└── theta_values.json

```

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) for dependency management

---

### Step 1: Clone the repository

```bash
git clone <repository_url>
cd <repository_name>
```

### Step 2: Configure the environment

Run the provided shell script to install dependencies and set up the project:

```bash
bash setup_project.sh
```

### Step 3: Enter Poetry Shell

Activate the Poetry shell to work within the project's virtual environment. This ensures that all commands use the correct dependencies:

```bash
poetry shell
```

**Why use the Poetry shell?**

- It activates a virtual environment where all project dependencies are isolated, preventing conflicts with global packages.
- Commands like `pytest` and `python` will directly reference the environment set up by Poetry.

### **Step 4: Train the Model**

To train the program with the provided dataset and visualize debugging plots, use the following command:

```bash
python3 -m linear_regression.train --plot_debug ./assets/data.csv
```

### **Step 5: Verify the Setup**

To ensure the setup is complete, run the test suite to verify the functionality:

```bash
poetry run pytest tests
```
## French version

# ft_linear_regression
## Index
- [Objectif du projet](#objectif-du-projet)
- [Analogie simplifiée pour comprendre le modèle d'entraînement](#analogie-simplifiée-pour-comprendre-le-modèle-dentraînement)
- [Concepts de base](#concepts-de-base)
  - [Fonction linéaire](#fonction-linéaire)
  - [Normalisation des données](#normalisation-des-données)
  - [Algorithme de descente de gradient](#algorithme-de-descente-de-gradient)
  - [Calcul des prédictions et des erreurs](#calcul-des-prédictions-et-des-erreurs)
  - [Mise à jour des paramètres](#mise-à-jour-des-paramètres)
  - [Calcul du coût](#calcul-du-coût)
  - [Vérification de la convergence](#vérification-de-la-convergence)
  - [Résultat final](#résultat-final)
- [Étapes](#étapes)
- [Structure du projet](#structure-du-projet)
- [Instructions de configuration](#instructions-de-configuration)
  - [Prérequis](#prérequis)
  - [Étape 1 : Cloner le dépôt](#étape-1--cloner-le-dépôt)
  - [Étape 2 : Configurer l'environnement](#étape-2--configurer-lenvironnement)
  - [Étape 3 : Entrer dans le shell Poetry](#étape-3--entrer-dans-le-shell-poetry)
  - [Étape 4 : Entraîner le modèle](#étape-4--entraîner-le-modèle)
  - [Étape 5 : Vérifier la configuration](#étape-5--vérifier-la-configuration)

L'objectif de ce projet est d'entraîner un modèle de régression linéaire pour prédire des résultats basés sur des données d'entrée, de comprendre les fondements mathématiques du modèle et d'évaluer ses performances.

Nous allons utiliser l'algorithme de descente de gradient ainsi que la fonction de régression linéaire pour entraîner le modèle et trouver la meilleure prédiction basée sur les données d'entrée (points de données).

## Analogie simplifiée pour comprendre le modèle d'entraînement

Imaginez que vous descendez une montagne pour trouver le point le plus bas (le meilleur ajustement pour les données) :

1. Vous commencez à un endroit aléatoire sur la montagne (point de départ $\theta_0$ et pente initiale $\theta_1$).
2. Vous cherchez la direction la plus abrupte vers le bas (le gradient).
3. Vous faites de petits pas (contrôlés par le taux d'apprentissage).
4. Vous ajustez votre direction à chaque pas pour continuer à descendre (mise à jour des paramètres).
5. Vous évaluez régulièrement à quel point vous êtes proche du bas (calcul du coût).
6. Vous vous arrêtez lorsque le terrain devient presque plat (le coût ne change plus beaucoup).
7. Bravo ! Vous avez trouvé la meilleure ligne d'ajustement.

## Concepts de base

### Fonction linéaire

Une **fonction linéaire** est un type simple de fonction où la relation entre l'entrée $(x)$ et la sortie $(y)$ est une ligne droite lorsque vous la tracez sur un graphique.

Elle suit cette formule :

$$
\large \hat{y} = \theta_0 + \theta_1 x
$$

**Où** :
- $θ_0$ (ordonnée à l'origine) : Là où la ligne croise l'axe des y (le point de départ de la ligne).
- $θ_1$ (pente) : L'inclinaison de la ligne.
- $x$ : Valeur d'entrée (par exemple, le kilométrage).
- $δy$ : Sortie prédite (par exemple, le prix).

**Explication graphique**

Pour une compréhension visuelle, vous pouvez consulter cette [explication graphique](https://www.notion.so/jvalenci/Resolve-question-I-don-t-understand-why-in-this-exercise-we-use-two-thetas-instead-of-jus-1-1599d52658e08088b639c429ed2d8311).

---

### Normalisation des données

**La normalisation** en science des données consiste à mettre vos données à une échelle standard (par exemple, de 0 à 1). Cela aide les modèles de régression linéaire à mieux fonctionner en s'assurant que toutes les caractéristiques contribuent également au résultat.

**La dénormalisation** est le processus inverse, où vous convertissez les données mises à l'échelle en leur échelle originale. Cela facilite l'interprétation des prédictions du modèle dans le contexte des données d'origine.

_Exemple :_

- **Normalisation** : Convertir des âges allant de 0 à 100 en une plage de 0 à 1.
- **Dénormalisation** : Reconvertir les âges prédits normalisés en une plage de 0 à 100.

**Formule de normalisation** :

$$
x_{\text{norm}} = \frac{x - \mu_x}{\sigma_x}, \quad y_{\text{norm}} = \frac{y - \mu_y}{\sigma_y}
$$

- $μ_x$ et $μ_y$ : Moyennes de $x$ et $y$.
- $σ_x$ et $σ_y$ : Écarts-types de $x$ et $y$.

**Ligne de régression sur des données normalisées** :

$$
y_{\text{norm}} = \theta_0 + \theta_1 \cdot x_{\text{norm}}
$$

**Revenir à l'échelle originale** :

- **Pente (sortie - inclinaison)** :
$$
\text{pente} = \theta_1 \cdot \frac{\sigma_y}{\sigma_x}
$$

- **Ordonnée à l'origine (entrée - départ de la ligne)** :
$$
\text{intercept} = \mu_y - \text{pente} \cdot \mu_x
$$

**Ligne de régression finale** :
$$
\hat{y} = \text{intercept} + \text{pente} \cdot x
$$

**Formule complète** :
$$
\hat{y} = (\theta_0 \cdot \sigma_y + \mu_y) + (\theta_1 \cdot \frac{\sigma_y}{\sigma_x}) \cdot (x - \mu_x)
$$

---
### Algorithme de descente de gradient

**Qu'est-ce que la descente de gradient ?**
La descente de gradient revient à trouver le point le plus bas d'une colline. Vous partez d'un endroit aléatoire sur la colline et descendez petit à petit jusqu'à atteindre le bas.

**Objectif**
Son objectif est de trouver le point le plus bas (le minimum) d'une "fonction de coût". Ce point le plus bas rend les prédictions de votre modèle aussi précises que possible.

**Formule** :

$$
\theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

$$
\theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i
$$

- **$\theta_0, \theta_1$** : Paramètres (ordonnée à l'origine et pente).
- **$\alpha$** : Taux d'apprentissage.
- **$m$** : Nombre de points de données.
- **$\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)$** : Erreur moyenne.
- **$\frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i$** : Erreur moyenne pondérée.

---

### Calcul des prédictions et des erreurs

**Objectif** :
- Se concentre sur le calcul du gradient (la pente de la fonction de coût) pour mettre à jour $\theta_0$ et $\theta_1$ à chaque itération de la descente de gradient.

**Concept mathématique** :
- Prédiction pour chaque point de données $(x_i, y_i)$ :
$$
  \hat{y}_i = \theta_0 + \theta_1 x_i
$$
- Erreur pour chaque point de données :
$$
  e_i = \hat{y}_i - y_i
$$

**Code** :
```python
for i in range(m):
    predicted = theta0 + theta1 * x[i]
    error = predicted - y[i]
    tmp_theta0 += error
    tmp_theta1 += error * x[i]
```

**Explication parallèle** :
- **Dans le code** : `predicted = theta0 + theta1 * x[i]` reflète $\hat{y}_i = \theta_0 + \theta_1 x_i$.
- **Dans le code** : `error = predicted - y[i]` correspond à $e_i = \hat{y}_i - y_i$.
- **Dans le code** : Accumuler `tmp_theta0 += error` et `tmp_theta1 += error * x[i]` correspond à additionner tous les $e_i$ et $e_i x_i$ pour les points de données :

$$
  \frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \quad\text{et}\quad \frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)x_i
$$

---

### Mise à jour des paramètres
**Objectif** :
La descente de gradient ajuste $\theta_0$ et $\theta_1$ pas à pas pour réduire l'erreur entre les valeurs prédites $\hat{y}$ et les valeurs réelles $y$. Elle utilise un petit pas d'apprentissage basé sur les données pour trouver la meilleure ligne d'ajustement.

**Concept mathématique** :
- Mise à jour de $\theta_0$ et $\theta_1$ en utilisant la descente de gradient :
  
$$
  \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

$$
  \theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i) x_i
$$

**Code** :
```python
theta0 -= (learning_rate / m) * tmp_theta0
theta1 -= (learning_rate / m) * tmp_theta1
```

---

### Calcul du coût

**Objectif** :
- `compute_cost` évalue dans quelle mesure le modèle actuel (avec des $θ_0$ et $θ_1$ spécifiques) correspond aux données, en fournissant une mesure numérique de l'erreur (coût).

**Concept mathématique** :
- Le coût (erreur quadratique moyenne) :
$$
  J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

**Code** :
- **Dans le code** : `compute_cost` calcule $\frac{1}{2m}\sum ( \hat{y}_i - y_i )^2$.

---
### Vérification de la convergence

- La convergence signifie que les paramètres ne réduisent plus significativement le coût :
- **Objectif** : L'objectif de la vérification de la convergence est d'arrêter la descente de gradient lorsque d'autres itérations n'améliorent plus significativement la précision, ce qui permet d'économiser des ressources et du temps.

**Concept mathématique** :

$$
\frac{|J_{\text{précédent}} - J_{\text{actuel}}|}{J_{\text{précédent}}} < \text{tolérance}
$$

**Paramètres de la formule de convergence** :

- **J_précédent** : Le coût de l'itération précédente.
- **J_actuel** : Le coût de l'itération actuelle.
- **tolérance** : Une petite valeur seuil pour déterminer si le changement de coût est négligeable.

**Code** :
```python
current_tolerance = abs((previous_cost - cost) / previous_cost)  # Tolérance relative
print(f"Iteration {it}: Cost = {cost:.6f}, Relative Tolerance = {current_tolerance:.6f}")
if current_tolerance < tolerance:  # Vérifier si l'amélioration relative est en dessous du seuil
  print(f"Convergence reached at iteration {it}. Final Cost = {cost:.6f}")
  break

previous_cost = cost
```

**Explication parallèle** :
- **Dans le code** : Si l'amélioration relative du coût est inférieure à `tolerance`, nous arrêtez.
- **En mathématiques** : Lorsque les changements de $J(\theta_0, \theta_1)$ deviennent négligeables par rapport à leur valeur actuelle, cela indique que la solution a (environ) convergé.

---

### Résultat final

* **Concept** : Après un nombre suffisant d'itérations, la ligne est aussi proche que possible du meilleur ajustement. Retournez la pente et l'ordonnée à l'origine finales ($\theta_1$ et $\theta_0$

