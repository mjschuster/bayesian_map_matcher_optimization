#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

def get_observations(X):
    return f(X).ravel()

def get_next_samples_from_user():
    input_string = input("Input a sequence of floats for x-values which should be observed next; Space separated, e.g. 1 3.2 .3 5.2:\n")
    if input_string == "":
        return None
    try:
        input_list = [float(f) for f in input_string.split(' ')]
    except ValueError as e:
        print(e, ". Ignoring this input.")
        return []
    return input_list
            
# Setup
plt.ion()
observations = ([], []) # observations[0] are the parameters, observations[1] the targets
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

while True:
    # Get observations
    new_observations = get_next_samples_from_user()
    # Stop looping if no numbers were put in
    if new_observations is None:
        break;
    observations[0].extend(new_observations)
    observations[1].extend(f(new_observations))

    # Fitting
    gp.fit(np.atleast_2d(observations[0]).T, observations[1])

    # Prediction
    renderspace_x = np.atleast_2d(np.linspace(0, 10, 1000)).T

    y_pred, sigma = gp.predict(renderspace_x, return_std=True)

    # Plotting
    plt.close()
    fig = plt.figure()
    plt.plot(renderspace_x, f(renderspace_x), 'r:', label=u'$f(x) = x\, \sin(x)$')
    plt.plot(observations[0], observations[1], 'r.', markersize=10, label=u'Observations')
    plt.plot(renderspace_x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([renderspace_x, renderspace_x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()

## ----------------------------------------------------------------------
## now the noisy case
#X = np.linspace(0.1, 9.9, 500)
#X = np.atleast_2d(X).T
#
## Observations and noise
#y = f(X).ravel()
#dy = 0.5 + 1.0 * np.random.random(y.shape)
#noise = np.random.normal(0, dy)
#y += noise
#
## Instanciate a Gaussian Process model
#gp = GaussianProcessRegressor(kernel=kernel, alpha=(dy / y) ** 2,
#                              n_restarts_optimizer=10)
#
## Fit to data using Maximum Likelihood Estimation of the parameters
#gp.fit(X, y)
#
## Make the prediction on the meshed x-axis (ask for MSE as well)
#y_pred, sigma = gp.predict(x, return_std=True)
#
## Plot the function, the prediction and the 95% confidence interval based on
## the MSE
#fig = plt.figure()
#plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
#plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
#plt.plot(x, y_pred, 'b-', label=u'Prediction')
#plt.fill(np.concatenate([x, x[::-1]]),
#         np.concatenate([y_pred - 1.9600 * sigma,
#                        (y_pred + 1.9600 * sigma)[::-1]]),
#         alpha=.5, fc='b', ec='None', label='95% confidence interval')
#plt.xlabel('$x$')
#plt.ylabel('$f(x)$')
#plt.ylim(-10, 20)
#plt.legend(loc='upper left')
#
#plt.savefig('bar.svg')
