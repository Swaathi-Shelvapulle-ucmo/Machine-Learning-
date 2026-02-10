# Linear Regression: Normal Equation vs Gradient Descent

### Student Name : Swaathi Shelvapulle

This project demonstrates **Linear Regression** implemented in two different ways:

1. **Closed-form solution (Normal Equation)**
2. **Iterative optimization using Gradient Descent**

A synthetic dataset is generated, both approaches are applied, and their results are compared visually and numerically.

---

## Dataset

- **Input (`x`)**: 200 random values sampled uniformly between 0 and 5  
- **Target (`y`)**:  
  y=3+4x+ϵ where ϵ is Gaussian noise

- Gaussian noise is added to simulate real-world data imperfections.

A bias term (column of ones) is added to the input matrix to learn the intercept.

---

## Methods

### 1. Normal Equation
The optimal parameters are computed analytically using:

  <img width="468" height="55" alt="image" src="https://github.com/user-attachments/assets/4be18f9e-56de-432b-80e3-7d61be47cdf2" />

This gives the exact solution when the matrix inverse exists.

### 2. Gradient Descent
An iterative approach that minimizes Mean Squared Error (MSE):

- Learning rate: `0.05`
- Iterations: `1000`
- Parameters are updated using the gradient of the loss function.

The loss value is recorded at every iteration to visualize convergence.

---

## Output

The program prints:
- Intercept and slope from both methods

It also generates two plots:
1. **Data with fitted regression lines**
2. **Gradient Descent loss curve (MSE vs iterations)**

---

## How to Run

```bash
python3 qs7.py
```

## Output Image
<img width="468" height="167" alt="image" src="https://github.com/user-attachments/assets/832d100c-3566-493f-bda1-c3eee625758d" />

