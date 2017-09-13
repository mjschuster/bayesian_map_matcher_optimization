#! /usr/bin/python3

import pickle
import sys
import os
import rosparam

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)

class Observation(object):
    USED_PARAMETERS = ['submap/descr_rad']
    def __init__(self, toplevel_directory):
        self.translation_errors = []
        self.rotation_errors = []
        self.params = None
        # DLR map matcher evaluation specific code to get the parameters
        for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
            for file_name in files:
                if file_name == "params.yaml":
                    self.params = rosparam.load_file(os.path.join(root, file_name))[0][0]
                    break
            if not self.params is None:
                break
        # DLR map matcher evaluation specific code to get the result-data
        for root, dirs, files in os.walk(toplevel_directory, topdown=False, followlinks=True):
            for file_name in files:
                if os.path.splitext(file_name)[1] == ".pkl":
                    pickle_path = os.path.join(root, file_name)
                    eval_result_data = pickle.load(open(pickle_path, 'rb'), encoding='latin1')
                    self.translation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['translation'].values())
                    self.rotation_errors.extend(eval_result_data['results']['hough3d_to_ground_truth']['rotation'].values())
        print("Got", len(self.translation_errors), "matches.")

    def get_gp_params(self):
        return [self.params[p] for p in self.USED_PARAMETERS]

    def get_gp_metric(self):
        return sum(self.translation_errors) / len(self.translation_errors)

def f(x):
    """The function to predict."""
    return x * np.sin(x)

def get_observations(X):
    return f(X).ravel()

def get_next_samples_from_user():
    input_string = input("Input a sequence of floats for param values for descr_rad:\n")
    if input_string == "":
        return None
    try:
        input_list = [float(f) for f in input_string.split(' ')]
    except ValueError as e:
        print(e, ". Ignoring this input.")
        return []
    return input_list
            
# Load data
observations = {}
observations_path = os.path.abspath(sys.argv[1])
for d in os.listdir(observations_path):
    current_observation_path = os.path.join(observations_path, d)
    observation_pickle_path = os.path.join(current_observation_path, "observation.pkl")
    if os.path.isfile(observation_pickle_path):
        print("Loading pickled observation from", observation_pickle_path)
        observation = pickle.load(open(observation_pickle_path, 'rb'))
    else:
        print("Creating observation from data in", current_observation_path)
        observation = Observation(current_observation_path)
        pickle.dump(observation, open(observation_pickle_path, 'wb'))
        print("Saved it to", observation_pickle_path)
        print(observation.get_gp_params(), "->", observation.get_gp_metric(), "| nr of matches:", len(observation.rotation_errors))
    observations[observation.get_gp_params()[0]] = [observation.get_gp_metric(), len(observation.rotation_errors)]
print("Available data looks like this: {descr_rad: [mean_err, nr_matches]}:\n", observations)
print("Available observations are:")
for obs_key in observations.keys():
    print(obs_key, end=" ")
print()

# Setup plots and gaussian process regressor
plt.ion()
gp_observations = ([], []) # observations[0] are the parameters, observations[1] the targets
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

while True:
    # Get observations
    new_observations = get_next_samples_from_user()
    # Stop looping if no numbers were put in
    if new_observations is None:
        break;
    # Add the observations given by the user to the datastructure the GP uses:
    for new_o in new_observations:
        # Check if we actually have that observation
        if new_o in observations.keys():
            if not new_o in gp_observations[0]:
                gp_observations[0].append(new_o)
                gp_observations[1].append(observations[new_o][0])
            else:
                print(new_o, "already observed by gp, skipping.")
        else:
            print("Observation value", new_o, "not available, skipping.")

    # Fitting
    gp.fit(np.atleast_2d(gp_observations[0]).T, gp_observations[1])

    # Prediction
    renderspace_x = np.atleast_2d(np.linspace(0, 0.7, 1000)).T

    y_pred, sigma = gp.predict(renderspace_x, return_std=True)

    # Plotting
    plt.close()
    fig = plt.figure()
    plt.plot(gp_observations[0], gp_observations[1], 'r.', markersize=10, label=u'Observations')
    plt.plot(renderspace_x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([renderspace_x, renderspace_x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('descr size (CSHOT)')
    plt.ylabel('mean translation error')
    plt.ylim(0, 0.4)
    plt.legend(loc='upper left')
    plt.show()
