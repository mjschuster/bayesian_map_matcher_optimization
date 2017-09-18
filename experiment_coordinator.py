#! /usr/bin/env python3
# Local imports
import evaluation_function
# import the DLR specific code to gather data
import dlr_map_matcher_interface_tools
INTERFACE_MODULE = dlr_map_matcher_interface_tools

# Foreign packages
import os
import rosparam
import pickle
import yaml

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
        # Set the python hash seed env variable to a fixed value, so hashing the rosparam dicts delivers the same results for the same dict on different python interpreter instances.
        os.environ['PYTHONHASHSEED'] = '42'

        ###########
        # Setup the SampleDatabase object
        ###########
        print("Initializing Sample database...")
        database_path = os.path.abspath(os.path.join(relpath_root, self._params['database_path']))
        sample_directory_path = os.path.abspath(os.path.join(relpath_root, self._params['sample_directory']))
        self.sample_db = SampleDatabase(database_path, sample_directory_path)

        ###########
        # Setup the evaluation function object
        ###########
        print("Initializing EvaluationFunction...")
        default_rosparams = rosparam.load_file(os.path.join(relpath_root, self._params['default_rosparams_yaml_path']))[0][0]
        optimized_rosparams = self._params['optimized_rosparams']
        #self.eval_function = EvaluationFunction(default_rosparams, self._params[..?) TODO

        ###########
        # Create an BayesianOptimization object, that contains the GPR logic.
        # Will supply us with new param-samples and will try to model the map matcher metric function.
        ###########
        # Put the information about which parameters to optimize in a form bayes_opt likes:
        print("Initializing Optimizer...")
        optimized_parameters = dict()
        for p_name, p_defs in self._params['optimized_rosparams'].items():
            optimized_parameters[p_defs['rosparam_name']] = (p_defs['min_bound'], p_defs['max_bound'])
            print("\tOptimizing parameter '", p_name, "' for optimization as '", p_defs['rosparam_name'],\
                  "' in compact set [", p_defs['min_bound'], ", ", p_defs['max_bound'], "]", sep="")
        #self.optimizer = BayesianOptimization(evaluate, optimized_parameters)

class SampleDatabase(object):
    """
    This object manages evaluation_function Samples in a dictionary (the "database").
    It knows nothing of the currently optimized parameters or their bounds, this functionality is covered in the evaluation_function module or the ExperimentCoordinator class.

    The databse-dict is indexed by the hashed parameters (hashed using the rosparam_hash function) and contains:
        * pickle_path: The path to a pickled evaluation_function Sample.
        * params: The complete rosparams dict used to generate this Sample. Its hash should be equal to the item's key.
    """

    def __init__(self, database_path, sample_dir_path):
        """
        Initializes a SampleDatabase object.

        :param database_path: Path to the database file; The pickled database dict.
        :param sample_dir_path: Path to the directory where samples created by this SampleDatabase should be stored.
        """
        # Error checking
        if os.path.isdir(database_path):
            raise ValueError("Given database path is a directory!", database_path)

        self._database_path = database_path
        self._sample_dir_path = sample_dir_path
        if os.path.exists(self._database_path): # If file exists...
            print("\tFound existing datapase pickle, loading from:", self._database_path, end=" ")
            # ...initialize the database from the file handle (hopefully points to a pickled dict...)
            with open(database_path, 'rb') as db_handle:
                self.db_dict = pickle.load(db_handle)
            print("- Loaded", len(self.db_dict), "samples.")
        else:
            print("\tDidn't find existing database pickle, initializing new database at", self._database_path, end=".\n")
            self.db_dict = {} # ..otherwise initialize as an empty dict and save it
            self.save()

    def save(self):
        """
        Pickles the current state of the database dict.
        """
        with open(self._database_path, 'wb') as db_handle:
            pickle.dump(self.db_dict, db_handle)

    def create_sample_from_map_matcher_results(self, results_path):
        """
        Creates a new Sample object from a finished map matcher run and adds it to the database.

        :param results_path: The path to the directory which contains the map matcher's results.
        """

        print("\tCreating new Sample from map matcher result at", results_path, end=".\n")
        sample = evaluation_function.Sample()
        INTERFACE_MODULE.create_evaluation_function_sample(results_path, sample)
        complete_rosparams = sample.parameters
        print("\tAdding sample at params", complete_rosparams)
        pickle_path = os.path.join(self._sample_dir_path, os.path.basename(results_path))
        self.db_dict[SampleDatabase.rosparam_hash(complete_rosparams)] = {'pickle_path' : pickle_path,
                                                                          'params' : complete_rosparams}
        self.save()

    def exists(self, complete_rosparams):
        """
        Returns whether a sample with the given rosparams already exists in the database.
        """

        return SampleDatabase.rosparam_hash(complete_rosparams) in self.db_dict.keys()

    def __getitem__(self, complete_rosparams):
        """
        Returns a evaluation_function Sample object from the database.

        :param complete_rosparams: The complete rosparams dictionary (not just the ones getting optimized).
        """

        # Get the sample
        params_hashed = SampleDatabase.rosparam_hash(complete_rosparams)
        extracted_sample = self.db_dict[params_hashed]
        # Do a sanity check of the parameters, just in case of a hash collision
        if not complete_rosparams == extracted_sample['params']:
            raise LookupError("Got a sample with hash " + params_hashed + ", but its parameters didn't match the requested parameters. (Hash function collision?)", complete_rosparams, extracted_sample['params'])
        return extracted_sample

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
        usage_string = "TODO" # TODO
        parser = argparse.ArgumentParser(description=usage_string)
        parser.add_argument('experiment_yaml',
                            help="Path to the yaml file which defines all parameters of one experiment run.")
        parser.add_argument('--add-sample', '-s',
                            dest='add_sample', default="",
                            help="Manually add one sample to the sample database and exit.")
        args = parser.parse_args()

        # Load the parameters from the yaml into a dict
        experiment_parameters_dict = yaml.safe_load(open(args.experiment_yaml))
        relpath_root = os.path.abspath(os.path.dirname(args.experiment_yaml))
        # Open the file handle to the database dict
        experiment_coordinator = ExperimentCoordinator(experiment_parameters_dict, relpath_root)
        if args.add_sample:
            print("--> Add sample mode <--")
            experiment_coordinator.sample_db.create_sample_from_map_matcher_results(args.add_sample)
        else:
            pass


    # Execute cmdline interface
    command_line_interface()
