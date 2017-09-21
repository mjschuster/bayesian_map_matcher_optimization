#! /usr/bin/env python3

# Local imports
import evaluation_function
# import the DLR specific code to gather data
import dlr_map_matcher_interface_tools
INTERFACE_MODULE = dlr_map_matcher_interface_tools

# Foreign packages
import matplotlib.pyplot as plt
import numpy as np
import os
import rosparam
import pickle
import yaml
import sys

from bayes_opt import BayesianOptimization

class ExperimentCoordinator(object):
    """
    Can be thought of as the 'toplevel' class for this repo.
    
    Will bring together the different parts of the system and manage the information flow
    between them.
    """

    def __init__(self, params_dict, relpath_root):
        """
        :param params_dict: A dictionary which contains all parameters needed to define an experiment.
        :param relpath_root: Basepath for all (other) relative paths in the params_dict.
        """
        self._params = params_dict
        self._relpath_root = relpath_root

        # Resolve relative paths
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
        self.sample_db = SampleDatabase(database_path, sample_directory_path, sample_generator_config)

        ###########
        # Setup the evaluation function object
        ###########
        print("Setting up EvaluationFunction...")
        optimization_definitions = self._params['optimization_definitions']
        default_rosparams = rosparam.load_file(rosparams_path)[0][0]
        self.eval_function = evaluation_function.EvaluationFunction(self.sample_db, default_rosparams,
                                                                    optimization_definitions,
                                                                    self._params['evaluation_function_metric'])
        ###########
        # Create an BayesianOptimization object, that contains the GPR logic.
        # Will supply us with new param-samples and will try to model the map matcher metric function.
        ###########
        # Put the information about which parameters to optimize in a form bayes_opt likes:
        print("Setting up Optimizer...")
        optimized_parameters = dict()
        for p_name, p_defs in self._params['optimization_definitions'].items():
            optimized_parameters[p_defs['rosparam_name']] = (p_defs['min_bound'], p_defs['max_bound'])
            print("\tOptimizing parameter '", p_name, "' as rosparam '", p_defs['rosparam_name'],\
                  "' in compact set [", p_defs['min_bound'], ", ", p_defs['max_bound'], "]", sep="")
        # Create the optimizer object
        self.optimizer = BayesianOptimization(self.eval_function.evaluate, optimized_parameters)
        # Get the initialization samples from the EvaluationFunction
        print("\tInitializing optimizer with", self._params['optimizer_initialization'])

    def initialize_optimizer(self):
        # init_dict will store the initialization data in the format the optimizer likes:
        # A list for each parameter with their values plus a 'target' list for the respective result value
        init_dict = {p_name: [] for p_name in self._params['optimizer_initialization'][0]}
        init_dict['targets'] = []
        # Fill init_dict:
        for optimized_rosparams in self._params['optimizer_initialization']:
            t = self.eval_function.evaluate(**optimized_rosparams)
            init_dict['targets'].append(t)
            for p_name, p_value in optimized_rosparams.items():
                init_dict[p_name].append(p_value)
        self.optimizer.initialize(init_dict)

    def save_gp_plot(self, plot_name):
        """
        Helper for saving plots of the GPR state to disk.
        """

        renderspace_x = np.atleast_2d(np.linspace(0, 1, 1000)).T
        y_pred, sigma = self.optimizer.gp.predict(renderspace_x, return_std=True)

        plt.plot(self.optimizer.X.flatten(), self.optimizer.Y, 'r.', markersize=10, label=u'Observations')
        plt.plot(renderspace_x, y_pred, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([renderspace_x, renderspace_x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(0, 1.2)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(self._params['plots_directory'], plot_name))
        plt.clf()

    def _resolve_relative_path(self, path):
        """
        Helper function for resolving paths relative to the _relpath_root-member.
        """
        if not os.path.isabs(path):
            return os.path.join(self._relpath_root, path)
        else:
            return path

class SampleDatabase(object):
    """
    This object manages evaluation_function Samples in a dictionary (the "database").
    It knows nothing of the currently optimized parameters or their bounds, this functionality is covered in the evaluation_function module.

    The databse-dict is indexed by the hashed parameters (hashed using the rosparam_hash function) and contains:
        * pickle_path: The path to a pickled evaluation_function Sample.
        * params: The complete rosparams dict used to generate this Sample. Its hash should be equal to the item's key.
    """

    def __init__(self, database_path, sample_dir_path, sample_generator_config):
        """
        Initializes a SampleDatabase object.

        :param database_path: Path to the database file; The pickled database dict.
        :param sample_dir_path: Path to the directory where samples created by this SampleDatabase should be stored.
        :param sample_generator_config: Config for the sample generation in the INTERFACE_MODULE.
        """
        # Error checking
        if os.path.isdir(database_path):
            raise ValueError("Given database path is a directory!", database_path)

        self._database_path = database_path
        self._sample_dir_path = sample_dir_path
        self._sample_generator_config = sample_generator_config
        if os.path.exists(self._database_path): # If file exists...
            print("\tFound existing datapase pickle, loading from:", self._database_path, end=" ")
            # ...initialize the database from the file handle (hopefully points to a pickled dict...)
            with open(database_path, 'rb') as db_handle:
                self._db_dict = pickle.load(db_handle)
            print("- Loaded", len(self._db_dict), "samples.")
        else:
            print("\tDidn't find existing database pickle, initializing new database at", self._database_path, end=".\n")
            self._db_dict = {} # ..otherwise initialize as an empty dict and save it
            self._save()

    def _save(self):
        """
        Pickles the current state of the database dict.
        """
        with open(self._database_path, 'wb') as db_handle:
            pickle.dump(self._db_dict, db_handle)

    def create_sample_from_map_matcher_results(self, results_path, override_existing=False):
        """
        Creates a new Sample object from a finished map matcher run and adds it to the database.

        :param results_path: The path to the directory which contains the map matcher's results.
        """

        print("\tCreating new Sample from map matcher result at", results_path, end=".\n")
        sample = evaluation_function.Sample()
        # This function actually fills the sample with data.
        # Its implementation depends on which map matching pipeline is optimized.
        INTERFACE_MODULE.create_evaluation_function_sample(results_path, sample)
        # Calculate the path where the new sample's pickle file should be placed
        pickle_basename = os.path.basename(results_path)
        if pickle_basename == "results": # In some cases the results are placed in a dir called 'results'
            # If that's the case, we'll use the name of the directory above, since 'results' is a bad name & probably not unique
            pickle_basename = os.path.basename(os.path.dirname(results_path))
        pickle_path = os.path.join(self._sample_dir_path, pickle_basename + ".pkl")
        # Safety check, don't just overwrite other pickles!
        if not override_existing and os.path.exists(pickle_path):
            raise ValueError("A pickle file already exists at the calculated location:", pickle_path)
        complete_rosparams = sample.parameters
        params_hashed = SampleDatabase.rosparam_hash(complete_rosparams)
        # Safety check, don't just overwrite a db entry
        if not override_existing and params_hashed in self._db_dict.keys():
            raise LookupError("Newly created sample's hash already exists in the database! Hash:", params_hashed,\
                              "Existing sample's pickle path is:", self._db_dict[params_hashed]['pickle_path'])
        print("\tPickling Sample object for later usage to:", pickle_path)
        self._pickle_sample(sample, pickle_path)
        # Add new Sample to db and save the db
        print("\tRegistering sample to database at hash(params):", params_hashed)
        self._db_dict[params_hashed] = {'pickle_path': pickle_path, 'params': complete_rosparams}
        self._save()

    def remove_sample(self, sample_identifier):
        """
        Removes a Sample's entry from the database and its pickled representation from disk.

        :param sample_identifier: A string which identifies the sample.
                                  Can either be a Sample's hash or
                                  the path where its pickled representation is stored on disk.
        """

        if os.path.isfile(sample_identifier): # sample_identifier points to the pickle file
            pickle_path = sample_identifier
            # Get the sample's representation
            sample = self._unpickle_sample(pickle_path)
            # Try to find and remove the Sample's entry in the database
            if not self.exists(sample.parameters):
                print("\tWarning: Couldn't find db-entry for pickled Sample '" + pickle_path + "'.")
            else:
                print("\tRemoving Sample's db entry '" + str(self.rosparam_hash(sample.parameters)) + "'.")
                del self._db_dict[SampleDatabase.rosparam_hash(sample.parameters)]
            # Remove it from disk
            print("\tRemoving Sample's pickle '" + pickle_path + "' from disk.")
            os.remove(pickle_path)
        else: # sample_identifier should be a hash in the db
            sample_hash = int(sample_identifier) # throws ValueError, if the file doesn't exist...
            if not sample_hash in self._db_dict:
                raise LookupError("Couldn't find a sample with hash", sample_hash)
            # Get sample from pickled representation
            pickle_path = self._db_dict[sample_hash]['pickle_path']
            if not os.path.isfile(pickle_path):
                print("\tWarning: Couldn't find Sample's pickled representation at '" +\
                      pickle_path + "'.")
            else:
                print("\tRemoving Sample's pickle '" + pickle_path + "' from disk.")
                os.remove(pickle_path)
            print("\tRemoving Sample's db entry '" + str(sample_hash) + "'.")
            del self._db_dict[sample_hash]
            
        # Only save the db at the end, after we know everything worked
        self._save()

    def exists(self, complete_rosparams):
        """
        Returns whether a sample with the given rosparams already exists in the database.
        """

        return SampleDatabase.rosparam_hash(complete_rosparams) in self._db_dict.keys()

    def __getitem__(self, complete_rosparams):
        """
        Returns a evaluation_function Sample object from the database.
        If the requested sample doesn't exist, it'll get generated.
        This causes the function call to block until it's done.

        :param complete_rosparams: The complete rosparams dictionary (not just the ones getting optimized).
        """

        params_hashed = SampleDatabase.rosparam_hash(complete_rosparams)
        # Check if we need to generate the sample first
        if not params_hashed in self._db_dict.keys():
            # Generate a new sample and store it in the database
            print("\tNo sample with hash ", params_hashed, " in databse, generating new sample:", sep="'")
            data_path = INTERFACE_MODULE.generate_sample(complete_rosparams, self._sample_generator_config)
            print("\tSample generation finished, adding it to database.")
            self.create_sample_from_map_matcher_results(data_path)
        # Get the sample's db entry
        sample_location = self._db_dict[params_hashed]['pickle_path']
        # load the Sample from disk
        print("\tRetrieved sample ", params_hashed, " from db, loading its representation from ", sample_location, sep="'")
        extracted_sample = self._unpickle_sample(sample_location)
        # Do a sanity check of the parameters, just in case of a hash collision
        if not complete_rosparams == extracted_sample.parameters:
            raise LookupError("Got a sample with hash " + params_hashed + ", but its parameters didn't match the requested parameters. (Hash function collision?)", complete_rosparams, extracted_sample.parameters)
        return extracted_sample

    def _pickle_sample(self, sample, pickle_path):
        """
        Helper method for saving samples to disk

        :param sample: The evaluation_function Sample which should be pickled.
        :param pickle_path: The path to where the Sample's pickled representation should be written to.
        """
        with open(pickle_path, 'wb') as sample_pickle_handle:
            pickle.dump(sample, sample_pickle_handle)

    def _unpickle_sample(self, pickle_path):
        """
        Helper method for loading samples from disk

        :param pickle_path: Path to the Sample's pickled representation.
        """
        with open(pickle_path, 'rb') as sample_pickle_handle:
            sample = pickle.load(sample_pickle_handle)
            if not isinstance(sample, evaluation_function.Sample):
                raise TypeError("The object unpickled from", pickle_path, "isn't an Sample instance!")
            return sample

    @classmethod
    def rosparam_hash(cls, params_dict):
        """
        Calculates and returns a hash from the given params_dict.
        """
        # Create a copy of the params dict and convert all lists to tuples. 
        # This is done, because lists aren't hashable.
        params_dict = params_dict.copy()
        for key, value in params_dict.items():
            if isinstance(value, list):
                params_dict[key] = tuple(value)
        return hash(frozenset(params_dict.items()))

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
        parser.add_argument('--remove-samples', '-rm',
                            dest='remove_samples', nargs='+', help=rm_arg_help)
        parser.add_argument('--add-samples', '-a',
                            dest='add_samples', nargs='+', help=add_arg_help)
        args = parser.parse_args()

        # Load the parameters from the yaml into a dict
        experiment_parameters_dict = yaml.safe_load(open(args.experiment_yaml))
        relpath_root = os.path.abspath(os.path.dirname(args.experiment_yaml))
        # Open the file handle to the database dict
        experiment_coordinator = ExperimentCoordinator(experiment_parameters_dict, relpath_root)

        # Check cmdline arguments for special modes, default mode (Experiment mode) is below
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

        print("--> Mode: Experiment <--")
        iteration = 0
        experiment_coordinator.initialize_optimizer()
        while True:
            experiment_coordinator.optimizer.maximize(init_points=0, n_iter=1, kappa=2)
            experiment_coordinator.save_gp_plot(str(iteration).zfill(5) + "_iteration.svg")
            iteration += 1

    # Execute cmdline interface
    command_line_interface()
