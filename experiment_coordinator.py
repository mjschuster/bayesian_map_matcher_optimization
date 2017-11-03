#! /usr/bin/env python3

# Local imports
import evaluation_function

# Foreign packages
import matplotlib.pyplot as plt
import numpy as np
import os
import rosparam
import yaml
import sys

from performance_measures import PerformanceMeasure

from bayes_opt import BayesianOptimization

class ExperimentCoordinator(object):
    """
    Can be thought of as the 'toplevel' class for this repo.
    Will bring together the different parts of the system and manage the information flow
    between them.
    
    Much code concerns reading / fixing paths, putting together the right initialization params for the used modules (mostly in __init__).
    Another big part is code for plotting results.
    """

    def __init__(self, params_dict, relpath_root):
        """
        :param params_dict: A dictionary which contains all parameters needed to define an experiment.
        :param relpath_root: Basepath for all (other) relative paths in the params_dict.
        """
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
        optimization_definitions = self._params['optimization_definitions']
        default_rosparams = rosparam.load_file(rosparams_path)[0][0]
        self.performance_measure = PerformanceMeasure.from_dict(self._params['performance_measure'])
        self.eval_function = evaluation_function.EvaluationFunction(self.sample_db, default_rosparams,
                                                                    optimization_definitions,
                                                                    self.performance_measure,
                                                                    self._params['rounding_decimal_places'])
        ###########
        # Create an BayesianOptimization object, that contains the GPR logic.
        # Will supply us with new param-samples and will try to model the map matcher metric function.
        ###########
        # Put the information about which parameters to optimize in a form bayes_opt likes:
        print("Setting up Optimizer...")
        optimized_rosparams = dict()
        for p_name, p_defs in self._params['optimization_definitions'].items():
            optimized_rosparams[p_defs['rosparam_name']] = (p_defs['min_bound'], p_defs['max_bound'])
            print("\tOptimizing parameter '", p_name, "' as rosparam '", p_defs['rosparam_name'],\
                  "' in compact set [", p_defs['min_bound'], ", ", p_defs['max_bound'], "]", sep="")
        # Create the optimizer object
        self.optimizer = BayesianOptimization(self.eval_function.evaluate, optimized_rosparams)

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

    def _setup_metric_axis(self, axis, param_name=None):
        """
        Helper method which sets up common properties of an axis object.
        """
        if param_name is None:
            param_name = list(self.eval_function.optimization_bounds)[0] # Get the first key
        axis.set_xlim(self.eval_function.optimization_bounds[param_name])
        axis.set_ylim(0,1)
        axis.set_xlabel(param_name)
        axis.set_ylabel(str(self.performance_measure))
        axis.yaxis.label.set_color('blue')
        axis.tick_params(axis='y', colors='blue')

    def plot_visualize_metric1d(self, plot_name, param_name):
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
        del(fixed_params[self.eval_function._optimization_definitions[param_name]['rosparam_name']])
        for optimized_params_dict, y_dict in self.eval_function.samples_filtered(fixed_params):
            optimized_param_values.append(optimized_params_dict[param_name]['value'])
            y_metric.append(y_dict['metric'])
            sample = y_dict['sample']
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
        print("\tSaving metric plot to", path)
        fig.savefig(path)
        fig.clf()

    def plot_gpr_single_param(self, plot_name, param_name):
        """
        Saves a plot of the GPR estimate to disk.
        Visualizes the estimated function mean and 95% confidence area from the GPR.
        Additionally shows the actual values for *all* available samples of the current EvaluationFunction.
        """

        # Setup the figure and its axes
        fig, metric_axis = plt.subplots()
        self._setup_metric_axis(metric_axis)

        # x-values for where to plot the predictions of the gaussian process
        bounds = self.eval_function.optimization_bounds[param_name]
        renderspace_x = np.atleast_2d(np.linspace(bounds[0], bounds[1], 1000)).T
        # Get mean and sigma from the gaussian process
        y_pred, sigma = self.optimizer.gp.predict(renderspace_x, return_std=True)
        # Plot the observations given to the gaussian process
        metric_axis.plot(self.optimizer.X.flatten(), self.optimizer.Y, 'b.', markersize=10, label=u'Observations')
        # Plot the gp's prediction mean
        metric_axis.plot(renderspace_x, y_pred, 'b-', label=u'Prediction')
        # Plot the gp's prediction 'sigma-tube'
        metric_axis.fill(np.concatenate([renderspace_x, renderspace_x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')

        # Plot all evaluation function samples we have
        samples_x = []
        samples_y = []
        # Gather available x, y pairs in the above two variables
        fixed_params = self.eval_function.default_rosparams.copy()
        del(fixed_params[self.eval_function._optimization_definitions[param_name]['rosparam_name']])
        for optimized_params_dict, y_dict in self.eval_function.samples_filtered(fixed_params):
            samples_x.append(optimized_params_dict[param_name]['value'])
            samples_y.append(y_dict['metric'])
        metric_axis.scatter(samples_x, samples_y, c='r', label=u"All known samples")
        metric_axis.set_ylim(0, 1)

        metric_axis.legend(loc='lower right')

        # Save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        print("\tSaving gpr plot to", path)
        fig.savefig(path)
        fig.clf()

    def _resolve_relative_path(self, path):
        """
        Helper function for resolving paths relative to the _relpath_root-member.
        """
        if not os.path.isabs(path):
            return os.path.join(self._relpath_root, path)
        else:
            return path


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
            experiment_coordinator.plot_visualize_metric1d("metric_visualization.svg", param_name)
            sys.exit()

        print("--> Mode: Experiment <--")
        iteration = 0
        experiment_coordinator.initialize_optimizer()
        while True:
            experiment_coordinator.optimizer.maximize(init_points=0, n_iter=1, kappa=2)
            for param_name, param_dict in experiment_parameters_dict['optimization_definitions'].items():
                plot_name = param_name.replace(" ", "_") + "_" + str(iteration).zfill(5) + "_iteration.svg"
                experiment_coordinator.plot_gpr_single_param(plot_name, param_name)
            iteration += 1

    # Execute cmdline interface
    command_line_interface()
