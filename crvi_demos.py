r"""°°°
# CRVI Demos
(https://colab.research.google.com/github/cvxpy/cvxpy/blob/master/examples/machine_learning/ridge_regression.ipynb#scrollTo=ZkX0i-nBqJIJ)[original notebook]
(https://github.com/zotroneneis/machine_learning_basics/blob/master/bayesian_linear_regression.ipynb)[bayesian linear regression example]

In this notebook, we demonstrate 
°°°"""
# |%%--%%| <8RJpRuCLo9|Gdrbs3SGyw>

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# |%%--%%| <Gdrbs3SGyw|8ocqu4OeiF>
r"""°°°
### Writing the objective function

We can decompose the **objective function** as the sum of a **least squares loss function** and an $\ell_2$ **regularizer**.
°°°"""
# |%%--%%| <8ocqu4OeiF|GvBNb916TJ>

def loss_fn(X, Y, beta):
    return cp.pnorm(cp.matmul(X, beta) - Y, p=2)**2

def regularizer(beta):
    return cp.pnorm(beta, p=2)**2

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

# |%%--%%| <GvBNb916TJ|20eVfZLQQx>
r"""°°°
### Generating data
Because ridge regression encourages the parameter estimates to be small, and as such tends to lead to models with **less variance** than those fit with vanilla linear regression. We generate a small dataset that will illustrate this.
°°°"""
# |%%--%%| <20eVfZLQQx|5xlpNwrBup>

def generate_data(m=100, n=20, sigma=5):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    # Generate an ill-conditioned data matrix
    X = np.random.randn(m, n)
    # Corrupt the observations with additive Gaussian noise
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y

m = 100
n = 20
sigma = 5

X, Y = generate_data(m, n, sigma)
X_train = X[:50, :]
Y_train = Y[:50]
X_test = X[50:, :]
Y_test = Y[50:]

# |%%--%%| <5xlpNwrBup|x8f4aq7Mxg>
r"""°°°
### Fitting the model

All we need to do to fit the model is create a CVXPY problem where the objective is to minimize the the objective function defined above. We make $\lambda$ a CVXPY parameter, so that we can use a single CVXPY problem to obtain estimates for many values of $\lambda$.
°°°"""
# |%%--%%| <x8f4aq7Mxg|MGA1UulYCe>

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

lambd_values = np.logspace(-2, 3, 50)
train_errors = []
test_errors = []
beta_values = []
for v in lambd_values:
    lambd.value = v
    problem.solve()
    train_errors.append(mse(X_train, Y_train, beta))
    test_errors.append(mse(X_test, Y_test, beta))
    beta_values.append(beta.value)

# |%%--%%| <MGA1UulYCe|2iFQkXiIMO>
r"""°°°
### Evaluating the model

Notice that, up to a point, penalizing the size of the parameters reduces test error at the cost of increasing the training error, trading off higher bias for lower variance; in other words, this indicates that, for our example, a properly tuned ridge regression **generalizes better** than a least squares linear regression.
°°°"""
# |%%--%%| <2iFQkXiIMO|f3MOhUQV75>

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()

plot_train_test_errors(train_errors, test_errors, lambd_values)

# |%%--%%| <f3MOhUQV75|teplRskDhe>
r"""°°°
### Regularization path
As expected, increasing $\lambda$ drives the parameters towards $0$. In a real-world example, those parameters that approach zero slower than others might correspond to the more **informative** features. It is in this sense that ridge regression can be considered **model selection.**
°°°"""
# |%%--%%| <teplRskDhe|ORgMa1oroy>

def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = len(beta_values[0])
    for i in range(num_coeffs):
        plt.plot(lambd_values, [wi[i] for wi in beta_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()

plot_regularization_path(lambd_values, beta_values)
