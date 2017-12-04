#! /usr/bin/env python3

# Local imports
import evaluation_function
import performance_measures

# Foreign packages
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d # required for 3d plots
import numpy as np
import os
import rosparam
import yaml
import sys
import shutil # for removing full filetrees
import itertools
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
        # Stores all discovered "best samples" during the experiment as tuples of (iteration, sample)
        self.best_samples = list()
        self.max_performance_measure = 0 # Holds the currently best known performance measure score
        self._params = params_dict
        self._relpath_root = relpath_root

        if "rng_seed" in self._params.keys():
            # Set seed of numpy random number generator
            print("Setting RNG seed to", self._params["rng_seed"])
            np.random.seed(self._params["rng_seed"])

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
        self.performance_measure = performance_measures.PerformanceMeasure.from_dict(self._params['performance_measure'])
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
        self.optimizer = BayesianOptimization(self.eval_function.evaluate, opt_bounds, verbose=0)
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
        axis.set_ylim(self.performance_measure.value_range)
        axis.set_xlabel(param_display_name)
        axis.set_ylabel(str(self.performance_measure))
        axis.yaxis.label.set_color('blue')
        axis.tick_params(axis='y', colors='blue')

    def output_sampled_params_table(self):
        """
        Creates a markdown file sampled_params.md which contains a markdown-table of the sampled parameters.
        Resulting table is supposed to look something like this:
        | Param 1 | Param 2 | Param 3 |
        | -------:| -------:| -------:|
        |  0.2912 |   1.231 |   23.12 |
        |  0.1422 |   1.914 |   2.182 |
        |  0.7811 |   2.111 |   12.29 |
        Formatting may break in the current implementation, if parameter values are bigger (i.e. not rounded)
        then the longest display name size.
        """
        # get length of longest display name
        max_length = max([len(display_name) for display_name in self.optimization_defs.keys()])
        left_sep = "| "
        right_sep = " |"
        center_sep = " | "
        with open("sampled_params.md", 'w') as table_file:
            # Write table headers
            table_file.write(left_sep)
            for i, display_name in enumerate(self.optimization_defs.keys()):
                table_file.write(display_name.rjust(max_length, ' ')) # rjust fills string with spaces
                # write center or right separator, depending on whether we're at the last element
                table_file.write((center_sep if not i == len(self.optimization_defs.keys()) - 1 else right_sep))
            # Write table header separator
            table_file.write('\n' + left_sep)
            for i in range(len(self.optimization_defs)):
                # the colon position defines alignment of column text, in this case to the right
                table_file.write('-' * max_length + (":| " if not i == len(self.optimization_defs.keys()) - 1 else ":|"))
            # For each sample, create a row
            for x in self.optimizer.X:
                # Write sample's row
                table_file.write('\n' + left_sep)
                for i, display_name in enumerate(self.optimization_defs.keys()):
                    param_value = round(x[self._to_optimizer_id(display_name)], self._params['rounding_decimal_places'])
                    table_file.write(str(param_value).rjust(max_length, ' '))
                    # write center or right separator, depending on whether we're at the last element
                    table_file.write((center_sep if not i == len(self.optimization_defs.keys()) - 1 else right_sep))

    def plot_error_distribution(self, plot_name, sample, max_rotation_error, max_translation_error, iteration=None):
        """
        Saves a plot with the error distribution of the given sample as a violin plot.
        """
        if iteration is None:
            iteration = self.iteration
        fig, axes = plt.subplots(ncols=2)
        # Create the violin plots on the axes
        if sample.nr_matches > 1:
            v_axes = (axes[0].violinplot(sample.rotation_errors, points=100, widths=0.7, bw_method=0.5,
                                         showmeans=True, showextrema=True, showmedians=False),
                      axes[1].violinplot(sample.translation_errors, points=100, widths=0.7, bw_method=0.5,
                                         showmeans=True, showextrema=True, showmedians=False))
            for i, axis in enumerate(v_axes):
                plot_body = axis["bodies"][0]
                plot_body.set_facecolor('blue' if i == 0 else 'yellow')
                plot_body.set_edgecolor('black')
        else: # Special case for when only one match happened
            axes[0].scatter([1], sample.rotation_errors)
            axes[1].scatter([1], sample.translation_errors)
        # Set titles
        fig.suptitle("Iteration " + str(iteration) + ", " + str(sample.nr_matches) + " matches.")
        axes[0].set_title("Rotation Errors")
        axes[1].set_title("Translation Errors")
        # Set x-axis limits
        axes[0].set_yticks(np.linspace(0, max_rotation_error, 10))
        axes[1].set_yticks(np.linspace(0, max_translation_error, 10))
        # Save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path)
        print("\tSaved error distribution plot to", path)
        plt.close()

    def plot_best_samples_boxplots(self):
        fig, axes = plt.subplots(4, sharex=True, figsize=(10,12))
        fig.suptitle("Best Sample per Iteration", fontsize=16, fontweight='bold')
        # make a list of iterations, to be used as x-axis labels
        iterations = [best_sample_tuple[0] for best_sample_tuple in self.best_samples]
        # add another element for the inital params set
        iterations.append("initial")
        # Get inital params sample
        init_sample = self.initial_sample
        # make a list of corresponding values from 0 to n-1
        x_axis = range(len(iterations))
        # Setup the axis for the performance measure (in blue)
        axes[0].set_ylabel(str(self.performance_measure))
        axes[0].yaxis.label.set_color('blue')
        axes[0].tick_params(axis='y', colors='blue')
        axes[0].scatter(x_axis,
                        [self.performance_measure(best_sample_tuple[1]) for best_sample_tuple in self.best_samples] + [self.performance_measure(init_sample)],
                        color='blue')
        # Setup the axis for the number of matches (in red)
        axes[1].set_ylabel('Nr. of Matches')
        axes[1].yaxis.label.set_color('red')
        axes[1].tick_params(axis='y', colors='red')
        axes[1].ticklabel_format(useOffset=False) # Forbid offsetting y-axis values
        axes[1].scatter(x_axis,
                        [best_sample_tuple[1].nr_matches for best_sample_tuple in self.best_samples] + [init_sample.nr_matches],
                        color='red')
        # Setup the axis for the translation error boxplots (in magenta)
        axes[2].set_ylabel(u"$Err_{translation}$ [m]")
        axes[2].yaxis.label.set_color('m')
        axes[2].tick_params(axis='y', colors='m')
        axes[2].boxplot([best_sample_tuple[1].translation_errors for best_sample_tuple in self.best_samples] + [init_sample.translation_errors],
                        positions=x_axis)
        axes[2].set_xticks(x_axis, iterations)
        # Setup the axis for the rotation error boxplots (in cyan)
        axes[3].set_ylabel(u"$Err_{rotation}$ [deg]")
        axes[3].yaxis.label.set_color('c')
        axes[3].tick_params(axis='y', colors='c')
        axes[3].boxplot([best_sample_tuple[1].rotation_errors for best_sample_tuple in self.best_samples] + [init_sample.rotation_errors],
                        positions=x_axis)
        axes[3].set_xticks(x_axis, iterations)
        axes[3].set_xticklabels(iterations)
        axes[3].set_xlabel("iteration")
        # Save and close
        path = os.path.join(self._params['plots_directory'], "best_samples_boxplot.svg")
        fig.savefig(path)
        print("\tSaved boxplots of best samples to", path)
        plt.close()

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

    def plot_gpr_two_param_contour(self, plot_name, param_names):
        """
        Saves a 4 contour plot of the GPR to disk:
        One subplot for GPR mean, target function, GPR variance and the acquisiton function.
        The two axis will be the varied parameters, the color codes the respective values listed above.
        :param plot_name: Filename of the plot. (plot will be saved into the plots_directory of the experiment yaml)
                          If None, the plot will not be saved but instead be opened in interactive mode.
        :param param_names: List of names of the parameters that are shown in this plot.
        """
        fig, axes = plt.subplots(2, 2)
        RESOLUTION = 50 # determines the plots' RESOLUTION, i.e. how many values per dimension get sampled the GP
        OFFSET = 0.01 # offsets the x-, y-axes, so markers at the edge are visible
        # Prepare x and y data (the same for all plots)
        x_bounds = self.eval_function.optimization_bounds[self._to_rosparam(param_names[0])]
        x = np.linspace(x_bounds[0] - OFFSET, x_bounds[1] + OFFSET, RESOLUTION)
        y_bounds = self.eval_function.optimization_bounds[self._to_rosparam(param_names[1])]
        y = np.linspace(y_bounds[0] - OFFSET, y_bounds[1] + OFFSET, RESOLUTION)
        value_range = self.performance_measure.value_range
        # Set labels for all axes
        for ax in [sub_axes for sublist in axes for sub_axes in sublist]:
            ax.set_xlabel(param_names[0], fontsize=5)
            ax.set_ylabel(param_names[1], fontsize=5)
        # Set shared arguments for the contourplot method
        contour_kwargs = {'cmap': 'hot', 'extend': 'both'}
        ############
        # Prepare known samples plot
        samples_x, samples_y, samples_z = self._get_filtered_samples(param_names)
        if len(samples_x) > 0: # guard against crashes if no known samples exist in the plane we're currently looking at
            plot = axes[0][0].scatter(samples_x, samples_y, c=samples_z, cmap='hot', edgecolor='', vmin=value_range[0], vmax=value_range[1])
            # set x,y axis bounds, only needed for the scatter plot
            axes[0][0].set_xlim(x_bounds[0] - OFFSET, x_bounds[1] + OFFSET)
            axes[0][0].set_ylim(y_bounds[0] - OFFSET, y_bounds[1] + OFFSET)
            axes[0][0].set_title("All Known Samples")
            fig.colorbar(plot, ax=axes[0][0], ticks=np.linspace(value_range[0], value_range[1], 11), label=str(self.performance_measure))
        ############
        # Prepare mean plot
        # Get observations known to the GPR
        filtered_X, filtered_Y = self._get_filtered_observations(param_names)
        # Get the GPR's estimate data
        predictionspace = self._get_prediction_space(param_names, RESOLUTION)
        mean, sigma = self.optimizer.gp.predict(predictionspace, return_std=True)
        z_mean = np.reshape(mean, (RESOLUTION, RESOLUTION))
        plot = axes[0][1].contourf(x, y, z_mean, levels=np.linspace(value_range[0], value_range[1], RESOLUTION), **contour_kwargs)
        axes[0][1].set_title("Estimated Mean")
        fig.colorbar(plot, ax=axes[0][1], ticks=np.linspace(value_range[0], value_range[1], 11), label=str(self.performance_measure))
        # plot all observations the GPR has
        axes[0][1].scatter(filtered_X.T[self._to_optimizer_id(param_names[0])],
                           filtered_X.T[self._to_optimizer_id(param_names[1])], marker='+', edgecolor='white')
        ############
        # Prepare variance plot
        z_var = np.reshape(sigma * sigma, (RESOLUTION, RESOLUTION))
        plot = axes[1][0].contourf(x, y, z_var, levels=np.linspace(0, 0.15, RESOLUTION), **contour_kwargs)
        axes[1][0].set_title("Estimation Variance")
        fig.colorbar(plot, ax=axes[1][0], ticks=np.linspace(0, 0.5, 11), label=u"$\sigma^2$")
        # plot all observations the GPR has
        axes[1][0].scatter(filtered_X.T[self._to_optimizer_id(param_names[0])],
                           filtered_X.T[self._to_optimizer_id(param_names[1])], marker='+', edgecolor='white')
        ############
        # Prepare acquisiton function plot
        acq = self.optimizer.util.utility(predictionspace, gp=self.optimizer.gp,
                                          y_max=self.optimizer.res['max']['max_val']) 
        z_acq = np.reshape(acq, (RESOLUTION, RESOLUTION))
        plot = axes[1][1].contourf(x, y, z_acq, levels=np.linspace(0, 1, RESOLUTION), **contour_kwargs)
        axes[1][1].set_title("Acquisition Function")
        kappa = 2 if not 'optimizer_params' in self._params.keys() else self._params['optimizer_params']['kappa']
        acq_label = u"$\mu + " + str(kappa) + u"\sigma$" # TODO: Change if using different acquisiton functions
        fig.colorbar(plot, ax=axes[1][1], ticks=np.linspace(0, 1, 11), label=acq_label)
        # plot all observations the GPR has
        axes[1][1].scatter(filtered_X.T[self._to_optimizer_id(param_names[0])],
                           filtered_X.T[self._to_optimizer_id(param_names[1])], marker='+', edgecolor='white')
        # Call mpl's magic layouting method (elimates overlap etc.)
        plt.tight_layout()
        # save and close
        path = os.path.join(self._params['plots_directory'], plot_name)
        fig.savefig(path) # Save an image of the 3d plot (which, of course, only shows one specific projection)
        print("\tSaved contour plot to", path)
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
        RESOLUTION = 50
        predictionspace = self._get_prediction_space(param_names, RESOLUTION)
        mean, sigma = self.optimizer.gp.predict(predictionspace, return_std=True)
        filtered_X, filtered_Y = self._get_filtered_observations(param_names)
        ax_3d.scatter(xs = filtered_X.T[self._to_optimizer_id(param_names[0])],
                      ys = filtered_X.T[self._to_optimizer_id(param_names[1])],
                      zs = filtered_Y, c='blue', label=u'Observations', s=50)
        ax_3d.plot_wireframe(predictionspace[:,self._to_optimizer_id(param_names[0])].reshape((RESOLUTION, RESOLUTION)),
                             predictionspace[:,self._to_optimizer_id(param_names[1])].reshape((RESOLUTION, RESOLUTION)),
                             mean.reshape((RESOLUTION, RESOLUTION)), label=u'Prediction')

        # Plot all evaluation function samples we have
        samples_x, samples_y, samples_z = self._get_filtered_samples(param_names)
        ax_3d.scatter(samples_x, samples_y, samples_z, c='red', label=u"All known samples", s=5)
        # Add legend
        ax_3d.legend(loc='lower right')
        # set limits
        ax_3d.set_xlim(self.eval_function.optimization_bounds[self._to_rosparam(param_names[0])])
        ax_3d.set_ylim(self.eval_function.optimization_bounds[self._to_rosparam(param_names[1])])
        ax_3d.set_zlim=(self.performance_measure.value_range)
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

    def _get_prediction_space(self, free_params, RESOLUTION=1000):
        """
        Returns a predictionspace for the gaussian process.
        :param free_params: Parameters which are supposed to vary in the prediction space.
                            For all other dimensions, the respective parameter will only have the value from the initial rosparams.
        :param RESOLUTION: The RESOLUTION for the prediction (higher means better quality)
        """
        predictionspace = np.zeros((len(self.optimizer.keys), np.power(RESOLUTION, len(free_params))))
        for key in [fixed_param for fixed_param in self.optimization_defs.keys() if fixed_param not in free_params]: # Fill all fixed params with the default value
            predictionspace[self._to_optimizer_id(key)] = np.full((1,np.power(RESOLUTION, len(free_params))), self.eval_function.default_rosparams[self._to_rosparam(key)])
        # Handle free params
        free_param_bounds = [self.eval_function.optimization_bounds[self._to_rosparam(key)] for key in free_params]
        free_param_spaces = [np.linspace(bounds[0], bounds[1], RESOLUTION) for bounds in free_param_bounds]
        free_param_coordinates = np.meshgrid(*free_param_spaces)
        for i, key in enumerate(free_params):
            predictionspace[self._to_optimizer_id(key)] = free_param_coordinates[i].flatten()
        predictionspace = predictionspace.T
        return predictionspace

    def _get_filtered_samples( self, free_params_display_names):
        """
        Returns samples from the sample database, filtered by their fixed params values.
        All parameters in the default rosparam dict are fixed, except those in the free_params_display_names list.
        NOTE: Currently only really works for len(free_params_display_names) == 2...

        :param free_params_display_names: List of parameter display names which are not fixed.
        """
        # Those members will hold the data
        samples_x = []
        samples_y = []
        samples_z = []
        # Gather available x, y pairs in the above two variables
        fixed_params = self.eval_function.default_rosparams.copy()
        # delete the free parameters from it
        del(fixed_params[self._to_rosparam(free_params_display_names[0])])
        del(fixed_params[self._to_rosparam(free_params_display_names[1])])
        for params_dict, metric_value, sample in self.eval_function.samples_filtered(fixed_params):
            samples_x.append(params_dict[self._to_rosparam(free_params_display_names[0])])
            samples_y.append(params_dict[self._to_rosparam(free_params_display_names[1])])
            samples_z.append(metric_value)
        return samples_x, samples_y, samples_z

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
        print("\033[1;4;35mIteration", self.iteration, ":\033[0m")
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter, kappa=kappa, **self.gpr_kwargs)
        # Dump the gpr's state for later use (e.g. interactive plots)
        pickle.dump(self, open(os.path.join(self._params['plots_directory'], "experiment_state.pkl"), 'wb'))
        # plot this iteration's gpr state in 2d for all optimized parameters
        self.plot_all_single_param()
        # plot this iteration's gpr state in 3d for the first two parameters (only if there are more than one parameter)
        display_names = list(self._params['optimization_definitions'].keys())
        if len(display_names) > 1 and len(display_names) < 4: # don't plot all pairs of parameters when there are more then 4, it'll take too much time.
            self.plot_all_two_params()
        # Check if we found a new best parameter set
        if self.max_performance_measure < self.optimizer.res['max']['max_val']:
            print("\t\033[1;35mNew maximum found, outputting params and plots!\033[0m")
            self.max_performance_measure = self.optimizer.res['max']['max_val']
            # Dump the best parameter set currently known by the optimizer
            yaml.dump(self.max_rosparams, open("best_rosparams" + self.iteration_string() + ".yaml", 'w'))
            # store the best known sample in the best_samples dict, for boxplots
            self.best_samples.append((self.iteration, self.max_sample))
            self.plot_all_new_best_params()
        self.output_sampled_params_table() # output a markdown table with all sampled params
        # increase iteration counter
        self.iteration += 1

    def plot_all_new_best_params(self, plot_all_violin=False):
        """
        Plots all kinds of plots for visualizing how the best parameters evolved.

        :param plot_all_violin: Whether to plot all violin plots.
                                If false, only the violin plot of self.max_sample is created except when a new max error is found.
        """
        # Find max rotation and translation errors from all best samples
        max_rotation_error = max(max(s[1].rotation_errors) for s in self.best_samples)
        max_translation_error = max(max(s[1].translation_errors) for s in self.best_samples)
        # If the new best sample induced a new max rotation/translation error, recreate all previous violin plots.
        # Otherwise, the y-axis will jump around when comparing the violin plots, which makes them pretty useless.
        if max_rotation_error == max(self.max_sample.rotation_errors) or max_translation_error == max(self.max_sample.translation_errors):
            print("\tNew max error, replotting all violin plots.")
            plot_all_violin = True

        if not plot_all_violin:
            # violin plot of the new best sample
            plot_path = os.path.join(self._params['plots_directory'], "violin_plot" + self.iteration_string() + ".svg")
            self.plot_error_distribution(plot_path, self.max_sample, max_rotation_error, max_translation_error)
        else:
            print(self.best_samples)
            # plot violin plots for all best samples
            for iteration, sample in self.best_samples:
                plot_path = os.path.join(self._params['plots_directory'], "violin_plot" + self.iteration_string(iteration) + ".svg")
                self.plot_error_distribution(plot_path, sample, max_rotation_error, max_translation_error, iteration)
            # also (re-)plot the initial params distribution
            plot_path = os.path.join(self._params['plots_directory'], "violin_plot_initial.svg")
            self.plot_error_distribution(plot_path, self.initial_sample, max_rotation_error, max_translation_error, "initial")

        # create a new version of the boxplot which includes the new sample
        self.plot_best_samples_boxplots()

    def plot_all_two_params(self):
        """
        Plots all kinds of plots that visualize two parameters at once.
        The method does this for all pairs of optimized parameters.
        """
        display_names = list(self._params['optimization_definitions'].keys())
        for display_name_pairs in itertools.combinations(display_names, 2):
            two_param_plot_name_prefix = display_name_pairs[0].replace(" ", "_") + "_" + display_name_pairs[1].replace(" ", "_") + self.iteration_string() + ".svg"
            self.plot_gpr_two_param_3d("3d_" + two_param_plot_name_prefix, display_name_pairs)
            self.plot_gpr_two_param_contour("contour_" + two_param_plot_name_prefix, display_name_pairs)

    def plot_all_single_param(self):
        """
        Plots all kinds of plots that visualize a single parameter.
        The method does this for all optimized parameters.
        """
        display_names = list(self._params['optimization_definitions'].keys())
        for display_name in display_names:
            self.plot_gpr_single_param(display_name.replace(" ", "_") + self.iteration_string() + ".svg", display_name)

    def iteration_string(self, iteration=None):
        """
        Creates a string to identify an iteration, mainly used for naming plots.
        Returns a string of form "_DDDDD_iteration", with DDDDD being the iteration's number, with prefixed zeros, if necessary.

        :param iteration: The iteration number as an integer, or None. If None, the current iteration (self.iteration) is used.
        """
        if iteration is None:
            iteration = self.iteration
        return "_" + str(iteration).zfill(5) + "_iteration"

    @property
    def initial_sample(self):
        """
        Returns the sample of the initial parameterset.
        """
        return self.sample_db[self.eval_function.default_rosparams]

    @property
    def max_sample(self):
        """
        Returns the sample of the best known parameterset.
        """
        return self.sample_db[self.max_rosparams]

    @property
    def max_rosparams(self):
        """
        Returns the complete rosparams dict with the best known parameters set.
        Of course, only optimized parameters will potentially be different from the inital param set.
        """
        # Get the initial parameter values, including those which didn't get optimized
        best_rosparams = self.eval_function.default_rosparams.copy()
        # Get the best known optimized parameters as a dict from the optimizer
        best_optimized_params = self.optimizer.res['max']['max_params'].copy()
        # Fix parameter types and round its values
        self.eval_function.preprocess_optimized_params(best_optimized_params)
        # Update the initial params dict with optimized params dict
        best_rosparams.update(best_optimized_params)
        return best_rosparams


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
        parser.add_argument('--clean-map-matcher-env', '-cmme',
                            dest='clean_mme', action='store_true',
                            help="Removes all map matcher environment directories, which don't have a sample in the sample database associated with it." +\
                                 "I.e. remove all directories of map matcher runs that didn't finish and, because of that, weren't added to the database.")
        parser.add_argument('--add-samples', '-a',
                            dest='add_samples', nargs='+', help=add_arg_help)
        parser.add_argument('--plot-metric', '-p',
                            dest='plot_metric', nargs='+',
                            help="Plots a 1D visualization of the metric's behaviour when changing the given parameter." +\
                                 " (parameter name has to fit the optimization definitions in the yaml file)")
        parser.add_argument('--3d-plot', '-3d',
                            dest='plot3d', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and opens interactive figures from their states. " +\
                                  "This may require setting your backend to something that supports interactive mode." +\
                                  "(see ~/.config/matplotlib/matplotlibrc for example)")
        parser.add_argument('--plot-single', '-p1',
                            dest='plot_single', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and plots their contents in 'single' plots." +\
                                 "Have a look at ExperimentCoordinator's 'plot_all_single_param' method, for further information.")
        parser.add_argument('--plot-two', '-p2',
                            dest='plot_two', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and plots their contents in 'single' plots." +\
                                 "Have a look at ExperimentCoordinator's 'plot_all_two_params' method, for further information.")
        parser.add_argument('--plot-max', '-pmax',
                            dest='plot_max', nargs='+',
                            help="Loads the supplied experiment state pickle file(s) and plots their contents in 'single' plots." +\
                                 "Have a look at ExperimentCoordinator's 'plot_all_new_best_params' method, for further information.")
        args = parser.parse_args()

        if args.plot3d:
            print("--> Interactive 3D plot mode <--")
            for path in args.plot3d:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_gpr_two_param_3d(None, list(experiment_coordinator._params['optimization_definitions'].keys()))
            sys.exit()
        if args.plot_single:
            print("--> Plot mode (single) <--")
            for path in args.plot_single:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_all_single_param()
            sys.exit()
        if args.plot_two:
            print("--> Plot mode (two) <--")
            for path in args.plot_two:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_all_two_params()
            sys.exit()
        if args.plot_max:
            print("--> Plot mode (max) <--")
            for path in args.plot_max:
                experiment_coordinator = pickle.load(open(path, 'rb'))
                experiment_coordinator.plot_all_new_best_params(plot_all_violin=True)
            sys.exit()

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
            for x, y, s in experiment_coordinator.eval_function:
                count += 1
                print(s)
                print("\tOptimized Parameters:")
                for display_name in experiment_coordinator.optimization_defs.keys():
                    print("\t\t", display_name, "=", x[experiment_coordinator._to_rosparam(display_name)])
                print("\tMetric-value:", y)
                if isinstance(experiment_coordinator.performance_measure, performance_measures.MixerMeasure):
                    print("\t\terror measure", type(experiment_coordinator.performance_measure.error_measure), "=",
                          experiment_coordinator.performance_measure.error_measure(s),
                          "(weight", 1 - experiment_coordinator.performance_measure.matches_weight, ")")
                    print("\t\tnr. matches measure", type(experiment_coordinator.performance_measure.matches_measure), "=",
                          experiment_coordinator.performance_measure.matches_measure(s),
                          "(weight", experiment_coordinator.performance_measure.matches_weight, ")")
            print("Number of usable samples:", count)
            sys.exit()
        if args.clean_mme:
            print("--> Mode: Clean Up Map Matcher Environment <--")
            # List of all sample origins in our database
            sample_origins = [os.path.basename(os.path.dirname(sample.origin)) for sample in experiment_coordinator.sample_db]
            mme_path = experiment_coordinator.sample_db.sample_generator_config['environment']
            # List of all map matcher ens
            map_matcher_envs = [os.path.abspath(os.path.join(mme_path, path)) for path in os.listdir(mme_path)]
            print("Number of map matcher envs:", len(map_matcher_envs), "; Number of sample origins:", len(sample_origins))
            print("Generating list of map matcher envs which don't have a sample associated with it...")
            to_delete_list = [mme_path for mme_path in map_matcher_envs if not os.path.basename(mme_path) in sample_origins]
            for path in to_delete_list:
                print(path)
            print("Delete those", len(to_delete_list), "map matcher envs? (y/n)")
            if input() in ['y', 'Y', 'yes', 'Yes']:
                count = 0
                for path in to_delete_list:
                    count += 1
                    print("deleted", count, "of", len(to_delete_list), "directories.", end='\r')
                    shutil.rmtree(path)
                print("All unused envs deleted.")
            else:
                print("aborted.")
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

        print("--> Mode: Experiment <--")
        experiment_coordinator.initialize_optimizer()
        while True:
            experiment_coordinator.iterate()

    # Execute cmdline interface
    command_line_interface()
