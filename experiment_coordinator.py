#! /usr/bin/env python3

# Local imports
import evaluation_function
from performance_measures import PerformanceMeasure

# Foreign packages
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os
import rosparam
import yaml
import sys
from bayes_opt import BayesianOptimization

class ExperimentCoordinator(object):
    """
    Can be thought of as the 'toplevel' class for this repo.
    Will bring together the different parts of the system and manage the information flow between them.
    
    Much code concerns reading / fixing paths, putting together the right initialization params for the used modules (mostly in __init__).
    Another big part is code for plotting results.

    For the plotting and all user-interface-related manners, the parameters will be described by their "display_name", i.e. their key in the experiment's yaml file.
    The code in the evaluation_function module doesn't know about the display_name and only uses the rosparam_name.
    """

    def __init__(self, params_dict, relpath_root):
        """
        :param params_dict: A dictionary which contains all parameters needed to define an experiment.
        :param relpath_root: Basepath for all (other) relative paths in the params_dict.
        """
        self.iteration = 0 # Holds the current iteration's number
        self._params = params_dict
        self._relpath_root = relpath_root

        # Resolve relative paths
        self._params['plots_directory'] = self._resolve_relative_path(self._params['plots_directory'])
        database_path = self._resolve_relative_path(self._params['database_path'])
        sample_directory_path = self._resolve_relative_path(self._params['sample_directory'])
        rosparams_path = self._resolve_relative_path(self._params['default_rosparams_yaml_path'])
        sample_generator_config = self._params['sample_generator_config']
        sample_generator_config['evaluator_executable'] = self._resolve_relative_path(sample_generator_config['evaluator_executable'])
        sample_generator_config['environment'] = self._resolve_relative_path(sample_generator_config['environment'])
        sample_generator_config['dataset'] = self._resolve_relative_path(sample_generator_config['dataset'])

        ###########
        # Setup the SampleDatabase object
        ###########
        print("Setting up Sample database...")
        # Create the sample db
        self.sample_db = evaluation_function.SampleDatabase(database_path, sample_directory_path, sample_generator_config)

        ###########
        # Setup the evaluation function object
        ###########
        print("Setting up EvaluationFunction...")
        self.optimization_defs = self._params['optimization_definitions']
        default_rosparams = rosparam.load_file(rosparams_path)[0][0]
        opt_bounds = dict()
        for p_name, p_defs in self.optimization_defs.items():
            opt_bounds[p_defs['rosparam_name']] = (p_defs['min_bound'], p_defs['max_bound'])
            print("\tOptimizing parameter '", p_name, "' as rosparam '", p_defs['rosparam_name'],\
                  "' in compact set [", p_defs['min_bound'], ", ", p_defs['max_bound'], "]", sep="")
        self.performance_measure = PerformanceMeasure.from_dict(self._params['performance_measure'])
        self.eval_function = evaluation_function.EvaluationFunction(self.sample_db, default_rosparams,
                                                                    opt_bounds,
                                                                    self.performance_measure,
                                                                    self._params['rounding_decimal_places'])
        ###########
        # Create an BayesianOptimization object, that contains the GPR logic.
        # Will supply us with new param-samples and will try to model the map matcher metric function.
        ###########
        print("Setting up Optimizer...")
        # Create the optimizer object
        self.optimizer = BayesianOptimization(self.eval_function.evaluate, opt_bounds)
        # Create a kwargs member for passing to the maximize method (see iterate())
        # Those parameters will be passed to the GPR member of the optimizer
        self.gpr_kwargs = {'alpha': self._params['gpr_params']['observation_noise']
                                    if 'gpr_params' in self._params else 1e-10} # defaults to gpr's default value

    def initialize_optimizer(self):
        # Get the initialization samples from the EvaluationFunction
        print("\tInitializing optimizer at", self._params['optimizer_initialization'])
        # init_dict will store the initialization data in the format the optimizer likes:
        # A list for each parameter with their values plus a 'target' list for the respective result value
        init_dict = {p_name: [] for p_name in self._params['optimizer_initialization'][0]}
        # Fill init_dict:
        for optimized_rosparams in self._params['optimizer_initialization']:
            for p_name, p_value in optimized_rosparams.items():
                init_dict[p_name].append(p_value)
        self.optimizer.explore(init_dict)

    def _setup_metric_axis(self, axis, param_display_name):
        """
        Helper method which sets up common properties of an axis object.

        :param axis: The axis object that needs to be set up
        :param param_display_name: The display name of the parameter, as given in the experiment's yaml file.
        """
        axis.set_xlim(self.eval_function.optimization_bounds[self._to_rosparam(param_display_name)])
        axis.set_ylim(0,1)
        axis.set_xlabel(param_display_name)
        axis.set_ylabel(str(self.performance_measure))
        axis.yaxis.label.set_color('blue')
        axis.tick_params(axis='y', colors='blue')

    def plot_metric_visualization1d(self, plot_name, param_name):
        """
        Saves a plot to visualize the metric's behaviour.

        Contains data from all available samples of the current EvaluationFunction.
        Shows all raw datapoints as well as the calculated metric.

        Only supports visualizing one dimension of optimized parameters.
        """
        # Setup the figure and its axes
        fig, (nr_matches_axis, translation_err_axis, rotation_err_axis) = plt.subplots(3, sharex=True, sharey=False)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        metric_axes = [nr_matches_axis.twinx(), translation_err_axis.twinx(), rotation_err_axis.twinx()]
        # set x and y lim for metric axes; Will also imply same x_lim for other axes.
        nr_matches_axis.set_ylabel("Number of Matches")
        nr_matches_axis.yaxis.label.set_color('red')
        nr_matches_axis.tick_params(axis='y', colors='red')
        translation_err_axis.set_ylabel(u"$Err_{translation}$ [m]")
        translation_err_axis.yaxis.label.set_color('m')
        translation_err_axis.tick_params(axis='y', colors='m')
        rotation_err_axis.set_ylabel(u"$Err_{rotation}$ [deg]")
        rotation_err_axis.yaxis.label.set_color('c')
        rotation_err_axis.tick_params(axis='y', colors='c')
        rotation_err_axis.set_xlabel(param_name)

        # Plot all evaluation function samples we have
        optimized_param_values = []
        y_metric = []
        y_nr_matches = []
        y_rotation_err = []
        y_translation_err = []
        fixed_params = self.eval_function.default_rosparams.copy()
        del(fixed_params[self._to_rosparam(param_name)])
        for params_dict, metric_value, sample in self.eval_function.samples_filtered(fixed_params):
            optimized_param_values.append(params_dict[self._to_rosparam(param_name)])
            y_metric.append(metric_value)
            y_nr_matches.append(sample.nr_matches)
            if not sample.nr_matches == 0:
                y_translation_err.append(sum(sample.translation_errors) / sample.nr_matches)
                y_rotation_err.append(sum(sample.rotation_errors) / sample.nr_matches)
            else:
                y_translation_err.append(0)
                y_rotation_err.append(0)
        # Sort the lists before plotting
        temp_sorted_lists = sorted(zip(*[optimized_param_values, y_metric, y_nr_matches, y_rotation_err, y_translation_err]))
        optimized_param_values, y_metric, y_nr_matches, y_rotation_err, y_translation_err = list(zip(*temp_sorted_lists))
        for metric_axis in metric_axes:
            self._setup_metric_axis(metric_axis, param_name)
            metric_axis.plot(optimized_param_values, y_metric, 'b:')
        nr_matches_axis.plot(optimized_param_values, y_nr_matches, 'r.')
        translation_err_axis.plot(optimized_param_values, y_translation_err, 'm.')
        rotation_err_axis.plot(optimized_param_values, y_rotation_err, 'c.')

        # Save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path)
        print("\tSaved metric plot to", path)
        plt.close()

    def plot_gpr_two_param_3d(self, plot_name, param_names):
        """
        Saves a 3D plot of the GPR estimate to disk.
        The X,Y axes will be parameter values, the Z axis the performance measure.
        :param plot_name: Filename of the plot. (plot will be saved into the plots_directory of the experiment yaml)
                          If None, the plot will not be saved but instead be opened in interactive mode.
        :param param_names: List of names of the parameters that are shown in this plot.
        """
        fig = plt.figure()
        ax_3d = fig.add_subplot(111, projection='3d')

        # Get data from the GPR
        resolution = 50
        predictionspace = self._get_prediction_space(param_names, resolution)
        mean, sigma = self.optimizer.gp.predict(predictionspace, return_std=True)
        filtered_X, filtered_Y = self._get_filtered_observations(param_names)
        ax_3d.scatter(xs = filtered_X.T[self._to_optimizer_id(param_names[0])],
                      ys = filtered_X.T[self._to_optimizer_id(param_names[1])],
                      zs = filtered_Y, c='blue', label=u'Observations')
        ax_3d.plot_wireframe(predictionspace[:,self._to_optimizer_id(param_names[0])].reshape((resolution, resolution)),
                             predictionspace[:,self._to_optimizer_id(param_names[1])].reshape((resolution, resolution)),
                             mean.reshape((resolution, resolution)), label=u'Prediction')

        # Plot all evaluation function samples we have
        # Get parameter values from the default rosparams
        fixed_params = self.eval_function.default_rosparams.copy()
        # delete the free parameters from it
        del(fixed_params[self._to_rosparam(param_names[0])])
        del(fixed_params[self._to_rosparam(param_names[1])])
        # Those members will hold the data
        samples_x = []
        samples_y = []
        samples_z = []
        # Gather available x, y pairs in the above two variables
        for params_dict, metric_value, sample in self.eval_function.samples_filtered(fixed_params):
            samples_x.append(params_dict[self._to_rosparam(param_names[0])])
            samples_y.append(params_dict[self._to_rosparam(param_names[1])])
            samples_z.append(metric_value)
        ax_3d.scatter(samples_x, samples_y, samples_z, c='red', label=u"All known samples", s=5)
        # Add legend
        ax_3d.legend(loc='lower right')
        # set limits
        ax_3d.set_xlim(self.eval_function.optimization_bounds[self._to_rosparam(param_names[0])])
        ax_3d.set_ylim(self.eval_function.optimization_bounds[self._to_rosparam(param_names[1])])
        ax_3d.set_zlim=(0,1)
        # set labels
        ax_3d.set_xlabel(param_names[0])
        ax_3d.set_ylabel(param_names[1])
        ax_3d.set_zlabel(str(self.performance_measure))

        # Save, show, clean up fig
        if plot_name is None:
            plt.show()
        else:
            path = os.path.join(self._params['plots_directory'], plot_name)
            fig.savefig(path) # Save an image of the 3d plot (which, of course, only shows one specific projection)
            print("\tSaved 3d plot to", path)
        plt.close()

    def plot_gpr_single_param(self, plot_name, param_name):
        """
        Saves a plot of the GPR estimate to disk.
        Visualizes the estimated function mean and 95% confidence area from the GPR.
        Additionally shows the actual values for *all* available samples of the current EvaluationFunction.
        """

        # Setup the figure and its axes
        fig, metric_axis = plt.subplots()
        self._setup_metric_axis(metric_axis, param_name)

        predictionspace = self._get_prediction_space([param_name])
        # Get mean and sigma from the gaussian process
        y_pred, sigma = self.optimizer.gp.predict(predictionspace, return_std=True)
        # Plot the observations available to the gaussian process
        filtered_X, filtered_Y = self._get_filtered_observations((param_name))
        metric_axis.plot(filtered_X.T[self._to_optimizer_id(param_name)], filtered_Y, 'b.', markersize=10, label=u'Observations')
        # Plot the gp's prediction mean
        plotspace = predictionspace[:,self._to_optimizer_id(param_name)]
        metric_axis.plot(plotspace, y_pred, 'b-', label=u'Prediction')
        # Plot the gp's prediction 'sigma-tube'
        metric_axis.fill(np.concatenate([plotspace, plotspace[::-1]]),
                         np.concatenate([y_pred - 1.9600 * sigma,
                                        (y_pred + 1.9600 * sigma)[::-1]]),
                         alpha=.5, fc='b', ec='None', label='95% confidence interval')

        # Plot all evaluation function samples we have
        samples_x = []
        samples_y = []
        # Gather available x, y pairs in the above two variables
        fixed_params = self.eval_function.default_rosparams.copy()
        del(fixed_params[self._to_rosparam(param_name)])
        for params_dict, metric_value, sample in self.eval_function.samples_filtered(fixed_params):
            samples_x.append(params_dict[self._to_rosparam(param_name)])
            samples_y.append(metric_value)
        metric_axis.scatter(samples_x, samples_y, c='r', label=u"All known samples")

        metric_axis.legend(loc='lower right')

        # Save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path)
        print("\tSaved plot to", path)
        plt.close()

    def _get_prediction_space(self, free_params, resolution=1000):
        """
        Returns a predictionspace for the gaussian process.
        :param free_params: Parameters which are supposed to vary in the prediction space.
                            For all other dimensions, the respective parameter will only have the value from the initial rosparams.
        :param resolution: The resolution for the prediction (higher means better quality)
        """
        predictionspace = np.zeros((len(self.optimizer.keys), np.power(resolution, len(free_params))))
        for key in [fixed_param for fixed_param in self.optimization_defs.keys() if fixed_param not in free_params]: # Fill all fixed params with the default value
            predictionspace[self._to_optimizer_id(key)] = np.full((1,np.power(resolution, len(free_params))), self.eval_function.default_rosparams[self._to_rosparam(key)])
        # Handle free params
        free_param_bounds = [self.eval_function.optimization_bounds[self._to_rosparam(key)] for key in free_params]
        free_param_spaces = [np.linspace(bounds[0], bounds[1], resolution) for bounds in free_param_bounds]
        free_param_coordinates = np.meshgrid(*free_param_spaces)
        for i, key in enumerate(free_params):
            predictionspace[self._to_optimizer_id(key)] = free_param_coordinates[i].flatten()
        predictionspace = predictionspace.T
        return predictionspace

    def _get_filtered_observations(self, free_params_display_names):
        fixed_params_display_names = [p for p in self.optimization_defs.keys() if not p in free_params_display_names]
        usables = []
        for i, x in enumerate(self.optimizer.X):
            # For each sample, check if it's usable:
            usable = True
            for fixed_param in fixed_params_display_names:
                if not self.eval_function.default_rosparams[self._to_rosparam(fixed_param)] == x[self._to_optimizer_id(fixed_param)]:
                    usable = False
                    break
            if usable:
                usables.append(i)
        # Filters self.optimizer.X with indices in usables along axis 0
        return np.take(self.optimizer.X, usables, 0), np.take(self.optimizer.Y, usables)

    def _to_optimizer_id(self, display_name):
        """
        Takes a "display_name" (i.e. the name for a parameter as defined in the experiment's yaml file)
        and returns its id in the optimizer's X-array.
        """
        rosparam_name = self._to_rosparam(display_name)
        for i, key in enumerate(self.optimizer.keys):
            if key == rosparam_name:
                return i
        raise LookupError("Rosparam", rosparam_name, "is not in the optimizer's key list. Maybe it isn't one of the optimized parameters?")

    def _to_rosparam(self, display_name):
        """
        Takes a "display_name" (i.e. the name for a parameter as defined in the experiment's yaml file)
        and returns its respective rosparam_name
        """
        return self.optimization_defs[display_name]['rosparam_name']

    def _resolve_relative_path(self, path):
        """
        Helper function for resolving paths relative to the _relpath_root-member.
        """
        if not os.path.isabs(path):
            return os.path.join(self._relpath_root, path)
        else:
            return path

    def iterate(self):
        """
        Runs one iteration of the system
        """
        if 'optimizer_params' in self._params.keys():
            init_points = self._params['optimizer_params']['pre_iteration_random_points']
            n_iter = self._params['optimizer_params']['samples_per_iteration']
            kappa = self._params['optimizer_params']['kappa']
        else:
            init_points = 0
            n_iter = 1
            kappa = 2
        # String for identify this iteration
        iteration_string = "_" + str(self.iteration).zfill(5) + "_iteration.svg"
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa, **self.gpr_kwargs)
        # Dump the gpr's state for later use (e.g. interactive plots)
        pickle.dump(self.optimizer, open(os.path.join(self._params['plots_directory'], iteration_string + "optimizer.pkl"), 'wb'))

        # plot this iteration's gpr state in 2d for all optimized parameters
        display_names = list(self._params['optimization_definitions'].keys())
        for display_name in display_names:
            self.plot_gpr_single_param(display_name.replace(" ", "_") + iteration_string, display_name)
        # plot this iteration's gpr state in 3d for the first two parameters (only if there are more than one parameter)
        if len(display_names) > 1:
            two_param_plot_name = "3d_" + display_names[0].replace(" ", "_") + "_" + display_names[1].replace(" ", "_") + "_" + str(self.iteration).zfill(5) + "_iteration.svg"
            self.plot_gpr_two_param_3d(two_param_plot_name, display_names)

        # increase iteration counter
        self.iteration += 1


if __name__ == '__main__': # don't execute when module is imported
    import argparse # for the cmd-line interface

    def command_line_interface():
        """
        Commandline interface for using this module
        """
        description_string = ("Command line utility for running Bayesian Optimization experiments on"
                              " parameter sets of a feature-based map matching pipeline."
                              " The program's default mode is the Experiment Mode."
                              " It only requires passing a yaml file as experiment_yaml parameter."
                              " See the other arguments' descriptions for different modes."
                             )
        rm_arg_help = ("Enters Remove Samples mode:"
                       " Manually remove EvaluationFunction Samples from the database and from disk."
                       " (only the pickled Sample, not the map matcher results from which it was generated)"
                       " Samples can be identified via their hash or the pickle path."
                       " Exits after all supplied samples have been removed."
                      )
        add_arg_help = ("Enters Add Samples mode:"
                        " Manually add EvaluationFunction Samples to the database."
                        " Simply supply the paths to the directories from which INTERFACE_MODULE can create it."
                        " In this mode, existing Samples with the same pickle location on disk,"
                        " and existing entries in the database (same rosparams-hash) will be overwritten."
                        " Exits after adding all supplied samples."
                       )

        parser = argparse.ArgumentParser(description=description_string)
        parser.add_argument('experiment_yaml',
                            help="Path to the yaml file which defines all parameters of one experiment run.")
        parser.add_argument('--list-all-samples', '-la',
                            dest='list_all_samples', action='store_true',
                            help="Lists all samples available in the database and exits.")
        parser.add_argument('--list-samples', '-ls',
                            dest='list_samples', action='store_true',
                            help="Lists those samples in the database, which are relevant for this experiment and exits.")
        parser.add_argument('--remove-samples', '-rm',
                            dest='remove_samples', nargs='+', help=rm_arg_help)
        parser.add_argument('--add-samples', '-a',
                            dest='add_samples', nargs='+', help=add_arg_help)
        parser.add_argument('--plot-metric', '-p',
                            dest='plot_metric', nargs='+',
                            help="Plots a 1D visualization of the metric's behaviour when changing the given parameter." +\
                                 " (parameter name has to fit the optimization definitions in the yaml file)")
        parser.add_argument('--interactive', '-i',
                            dest='interactive',
                            help="Opens the supplied pickle file as a matplotlib figure in interactive mode." +\
                                  "This may require setting your backend to something that supports interactive mode." +\
                                  "(see ~/.config/matplotlib/matplotlibrc for example)")
        args = parser.parse_args()

        # Load the parameters from the yaml into a dict
        experiment_parameters_dict = yaml.safe_load(open(args.experiment_yaml))
        relpath_root = os.path.abspath(os.path.dirname(args.experiment_yaml))
        # Open the file handle to the database dict
        experiment_coordinator = ExperimentCoordinator(experiment_parameters_dict, relpath_root)

        # Check cmdline arguments for special modes, default mode (Experiment mode) is below
        if args.list_all_samples:
            print("--> Mode: List All Samples <--")
            for sample in experiment_coordinator.sample_db:
                print(sample)
            print("Total number of samples", len(experiment_coordinator.sample_db._db_dict))
            sys.exit()
        if args.list_samples:
            print("--> Mode: List Samples <--")
            count = 0
            for X, Y in experiment_coordinator.eval_function:
                count += 1
                print(Y['sample'])
                print("\tOptimized Parameters:")
                for name, defs in X.items():
                    print("\t\t", name, "=", defs['value'])
                print("\tMetric-value:", Y['metric'])
            print("Number of usable samples:", count)
            sys.exit()
        if args.remove_samples:
            print("--> Mode: Remove Samples <--")
            for sample_id in args.remove_samples:
                experiment_coordinator.sample_db.remove_sample(sample_id)
            sys.exit()
        if args.add_samples:
            print("--> Mode: Add Samples <--")
            for sample_path in args.add_samples:
                experiment_coordinator.sample_db.create_sample_from_map_matcher_results(sample_path, override_existing=True)
            sys.exit()
        if args.plot_metric:
            print("--> Mode: Plot Metric <--")
            param_name = ' '.join(args.plot_metric)
            print("\tFor parameter '", param_name, "'", sep="")
            experiment_coordinator.plot_metric_visualization1d("metric_visualization.svg", param_name)
            sys.exit()
        if args.interactive:
            print("--> Interactive plot mode <--")
            experiment_coordinator.optimizer = pickle.load(open(args.interactive, 'rb'))
            experiment_coordinator.plot_gpr_two_param_3d(None, list(experiment_coordinator._params['optimization_definitions'].keys()))
            sys.exit()

        print("--> Mode: Experiment <--")
        experiment_coordinator.initialize_optimizer()
        while True:
            experiment_coordinator.iterate()

    # Execute cmdline interface
    command_line_interface()
