"""
Contains the code for modelling the EvaluationFunction via discrete samples.

A Sample is defined by its full set of rosparams ("complete_rosparams"), since they're necessary to run
the map matcher.
A Sample contains resulting data from the map matcher evaluation with its complete_rosparams.

The Samples are managed by the SampleDatabase, which uses a hashing to quickly find the Sample for a given complete_rosparams.
If it doesn't exist, another Sample will be generated using the INTERFACE_MODULE. (this is where you can define some method for generating Sample data for your specific map matcher implementation)

The EvaluationFunction is the toplevel class for this repo. In it, "complete_rosparams" and "optimized_rosparams" are differentiated.
It acts as the interface for the optimizer, which only knows about the "optimized_rosparams" (subset of "complete_rosparams").
"""

import numpy as np
import os
import pickle

# import the DLR specific code to gather data
import dlr_map_matcher_interface_tools
INTERFACE_MODULE = dlr_map_matcher_interface_tools

class EvaluationFunction(object):
    """
    Models the evaluation function that is optimized by the gaussian process.

    In our use-case, feature-based map matcher evaluation, the evaluation function can't be defined analytically.
    Instead, this class models it as a set of Sample points for which the map matcher pipeline was evaluated.

    This class uses the SampleDatabase class to manage the Samples over which it is defined.
    This should reduce the time subsequent experiments will require, after a bunch of samples have already been generated.
    """

    METRICS = ['test', 'mean_error']
    """
    Available metrics:
        * test: Metric for testing, will circumvent the sample generation and just model f(x)=50x*sin(50x).
                Will only work as long as just one parameter is optimized.
        * mean_error: Will return mean error of a Sample's matches.
    """

    def __init__(self, sample_db, default_rosparams, optimization_definitions, used_metric, rounding_decimal_places=0):
        """
        Creates an EvaluationFunction object.
        
        :param sample_db: A sample database object.
        :param default_rosparams: A dict that contains all rosparams required for running the map matcher
                                  with the default values.
        :param optimization_definitions: A dict that contains the definitions which rosparams which are beeing optimized in this experiment and within which bounds.
                                         Expects a dict that contains one or multiple entires, each containing rosparam_name, min_bound and max_bound.
        :param used_metric: A string to tell which metric should be used. See EvaluationFunction.METRICS for all implemented metrics.
                            Some metrics may terminate the experiment, if your Sample instances don't contain the necessary data.
        :param rounding_decimal_places: The number of decimal places to which parameters of type float should be rounded to.
                                        If zero, no rounding will take place.
        """

        # error checking
        if not used_metric in EvaluationFunction.METRICS:
            raise ValueError("Unknown metric", used_metric, "possible values:", EvaluationFunction.METRICS)

        self._sample_db = sample_db
        self._default_rosparams = default_rosparams
        self._optimization_definitions = optimization_definitions
        self._used_metric = used_metric
        self._rounding_decimal_places = rounding_decimal_places

    def evaluate(self, **optimized_rosparams):
        """
        This method supplies the interface for the Optimizer.
        Calculates and returns the current metric ("y") at the given parameters ("X").

        Will convert the optimized_rosparams (from the Optimizer-world) to the complete_rosparams which define 
        a Sample in the map matching pipeline.
        This may also involve casting types of optimized_rosparams to the type the parameter has in the default_rosparams.

        This function may return quickly, if the sample already exists in the sample database.
        Otherwise, the call will block until the map matching pipeline has finished generating the requested sample.

        :param optimized_rosparams: A keyworded argument list.
        """

        print("Evaluating function at", optimized_rosparams.items())

        # Error handling
        # Iterate over the optimization definitions and check if the current request doesn't violate them
        for p_name, p_dict in self._optimization_definitions.items():
            if not p_dict['rosparam_name'] in optimized_rosparams:
                raise ValueError(p_dict['rosparam_name'] + " should get optimized, but wasn't in given dict of optimized parameters.", optimized_rosparams)
            p_value = optimized_rosparams[p_dict['rosparam_name']]
            if p_value > p_dict['max_bound']:
                raise ValueError(p_dict['rosparam_name'] + " value (" + str(p_value) + ") is over max bound (" +\
                                 str(p_dict['max_bound']) + ").")
            if p_value < p_dict['min_bound']:
                raise ValueError(p_dict['rosparam_name'] + " value (" + str(p_value) + ") is under min bound (" +\
                                 str(p_dict['min_bound']) + ").")
        # Iterate over the current request...
        for p_name, p_value in optimized_rosparams.items():
            # ...check if there are parameters in there, that shouldn't be optimized.
            if not p_name in [opt_param['rosparam_name'] for opt_param in self._optimization_definitions.values()]:
                raise ValueError(str(p_name) + " shouldn't get optimized, but was in given dict of optimized parameters.")
            # ...also cast their type to the type in the default rosparams dict. Otherwise, we may serialize
            # some high-precision numpy float class instead of having a built-in float value on the yaml, that
            # rosparams can actually read. Sadl, we'll lose some precision through that.
            if type(self._default_rosparams[p_name]) != type(p_value):
                print("\tWarning, casting parameter type", type(p_value), "of", p_name, "to", type(self._default_rosparams[p_name]))
                optimized_rosparams[p_name] = type(self._default_rosparams[p_name])(p_value)
            if self._rounding_decimal_places and isinstance(p_value, float):
                rounded_p_value = round(optimized_rosparams[p_name], self._rounding_decimal_places)
                print("\tWarning, rounding float value", p_name, ":", p_value, "->", rounded_p_value)
                optimized_rosparams[p_name] = rounded_p_value

        if self._used_metric == 'test':
            # test metric expects only one optimized parameter, so just get the first value in the dict
            x = [p_value for p_value in optimized_rosparams.values()][0]
            return 50*x * np.sin(50*x)

        # Create the full set of parameters by updating the default parameters with the optimized parameters.
        complete_rosparams = self._default_rosparams.copy()
        complete_rosparams.update(optimized_rosparams)
        # Get the sample from the db (this call blocks until the sample is generated, if it isn't in the db)
        sample = self._sample_db[complete_rosparams]
        # Calculate the metric
        return self.metric(sample)

    def __iter__(self):
        """
        Iterator for getting all available samples from the database, which define this EvaluationFunction.

        This means only those Samples are yielded, which have parameter values matching the ones in default_rosparams.
        Only parameter values of currently optimized parameters are allowed to differ.
        The filtering is done in the _defined_by function.
        
        Returns samples as a tuple (X, Y), with X: A dict of the optimized_rosparams; Contains the param's 'value'
                                                                                      and its 'rosparam_name'.
                                         , and with Y: A dict with the current metric's value at 'metric' and
                                                       the Sample object itself at 'sample'.
        """
        for sample in self._sample_db:
            if self._defined_by(sample.parameters):
                X = dict()
                for name, defs in self._optimization_definitions.items():
                    X[name] = {'rosparam_name': defs['rosparam_name'],
                               'value': sample.parameters[defs['rosparam_name']]}
                Y = {'metric': self.metric(sample),
                     'sample': sample}
                yield (X, Y)

    def _defined_by(self, complete_rosparams):
        """
        Returns whether the given complete_rosparams is valid for defining this EvaluationFunction.
        That's the case if all non-optimized parameters of complete_rosparams are equal to this EvaluationFunction's default_rosparams.
        """
        nr_of_optimized_params_found = 0
        for param, value in complete_rosparams.items():
            # Check if the current param is optimized
            if param in [d['rosparam_name'] for d in self._optimization_definitions.values()]:
                nr_of_optimized_params_found += 1
                continue # Move on to the next parameter
            else: # If it's not optimized
                # check whether it has the right value (equal to the one set in default_rosparams)
                if not value == self._default_rosparams[param]:
                    return False # return False immediately

        if not nr_of_optimized_params_found == len(self._optimization_definitions):
            raise LookupError("There are parameters in the optimization_definitions, which aren't in the sample's complete rosparams.", str(self._sample_db[complete_rosparams]))

        return True

    def metric(self, sample):
        """
        Returns the active metric's value for a given Sample object.
        Its concrete behaviour depends on the used_metric parameter given to the init method.

        :param sample: The Sample for which the metric should be calculated.
        """

        # Error handling
        if not isinstance(sample, Sample):
            raise ValueError("Given sample object isn't a Sample.", sample)

        if self._used_metric == 'mean_error':
            translation_error = sum(sample.translation_errors) / sample.nr_matches
            rotation_error = sum(sample.rotation_errors) / sample.nr_matches
            # Adjust value range with some values (ugly, TODO)
            translation_error = (translation_error - 0.2) / 0.8
            rotation_error = (rotation_error - 2) / 6
            value = 1 / (translation_error + rotation_error) # invert the measure so the max is useful (ugly, TODO)
            #print("Sample's result with metric '" + self._used_metric + "': ", value)
            return value

class Sample(object):
    """
    Represents one sample of the evaluation function.
    On the map matcher side, this is one run of the map matcher on the whole dataset.
    It contains all available data, though only parts of it may be used by the EvaluationFunction that 
    is actually optimized.
    The data fields are as follows:
        * translation_errors: A list of translation errors meters per match
        * rotation_errors: A list of rotation errors in degree per match
        *---> Translation error n and rotation error n are both expected to be the result of match n.
        * parameters: A dict of all rosparams used for the map matcher run
        * time: A datetime.timedelta object, expressing the time-cost of this Sample.
        * origin: A string which describes where that sample was generated. Probably should be a path to the map matcher results directory.
    """

    def __init__(self):
        """
        Creates a function sample object with empty data contents.
        Fill with available data by accessing its propteries.
        All data fields are initialized with None here, so the code will crash if you use
        metrics on samples which don't have the data they need. (instead of generating wrong results)
        """
        self.translation_errors = None
        self.rotation_errors = None
        self.parameters = None
        self.time = None
        self.origin = None

    @property
    def nr_matches(self):
        """
        The number of matches the map matcher made with this Sample's parameters.
        """
        return len(self.translation_errors)

    def __str__(self):
        """
        Small, readable representation of this Sample
        """
        return "Sample generated from " + self.origin + " (database-hash: " + str(SampleDatabase.rosparam_hash(self.parameters)) + ")"

class SampleDatabase(object):
    """
    This object manages Samples in a dictionary (the "database").
    It knows nothing of the currently optimized parameters or their bounds, this functionality is covered by the EvaluationFunction.
    The sample object themselves aren't kept in memory, but are stored on disk as pickle files.
    They'll get loaded into when requested via __getitem__ like this: sample_db_obj[rosparams_of_requested_sample].

    The databse-dict is indexed by the hashed parameters (hashed using the rosparam_hash function) and contains:
        * pickle_name: The name of the pickled Sample object.
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

        results_path = os.path.abspath(results_path)
        print("\tCreating new Sample from map matcher result at", results_path, end=".\n")
        sample = Sample()
        # This function actually fills the sample with data.
        # Its implementation depends on which map matching pipeline is optimized.
        INTERFACE_MODULE.create_evaluation_function_sample(results_path, sample)
        # Calculate the path where the new sample's pickle file should be placed
        pickle_basename = os.path.basename(results_path)
        if pickle_basename == "results": # In some cases the results are placed in a dir called 'results'
            # If that's the case, we'll use the name of the directory above, since 'results' is a bad name & probably not unique
            pickle_basename = os.path.basename(os.path.dirname(results_path))
        pickle_name = pickle_basename + ".pkl"
        self._pickle_sample(sample, pickle_name, override_existing)
        complete_rosparams = sample.parameters
        params_hashed = SampleDatabase.rosparam_hash(complete_rosparams)
        # Safety check, don't just overwrite a db entry
        if not override_existing and params_hashed in self._db_dict.keys():
            raise LookupError("Newly created sample's hash already exists in the database! Hash:", params_hashed,\
                              "Existing sample's pickle path is:", self._db_dict[params_hashed]['pickle_name'])
        # Add new Sample to db and save the db
        print("\tRegistering sample to database at hash(params):", params_hashed)
        self._db_dict[params_hashed] = {'pickle_name': pickle_name, 'params': complete_rosparams}
        self._save()

    def remove_sample(self, sample_identifier):
        """
        Removes a Sample's entry from the database and its pickled representation from disk.

        :param sample_identifier: A string which identifies the sample.
                                  Can either be a Sample's hash or
                                  the path where its pickled representation is stored on disk.
        """

        if os.path.splitext(sample_identifier)[1] == '.pkl': # sample_identifier points to the pickle file
            pickle_name = sample_identifier
            # Get the sample's representation
            sample = self._unpickle_sample(pickle_name)
            # Try to find and remove the Sample's entry in the database
            if not self.exists(sample.parameters):
                print("\tWarning: Couldn't find db-entry for pickled Sample '" + pickle_name + "'.")
            else:
                print("\tRemoving Sample's db entry '" + str(self.rosparam_hash(sample.parameters)) + "'.")
                del self._db_dict[SampleDatabase.rosparam_hash(sample.parameters)]
            # Remove it from disk
            print("\tRemoving Sample's pickle '" + pickle_name + "' from disk.")
            os.remove(os.path.join(self._sample_dir_path, pickle_name))
        else: # sample_identifier should be a hash in the db
            sample_hash = int(sample_identifier) # throws ValueError, if the file doesn't exist...
            if not sample_hash in self._db_dict:
                raise LookupError("Couldn't find a sample with hash", sample_hash)
            # Get sample from pickled representation
            pickle_name = self._db_dict[sample_hash]['pickle_name']
            pickle_path = os.path.join(self._sample_dir_path, pickle_name)
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

    def __iter__(self):
        """
        Iterator for getting all sample objects contained in the database.
        """
        for sample_hash, sample in self._db_dict.items():
            yield self._unpickle_sample(sample['pickle_name'])

    def __getitem__(self, complete_rosparams):
        """
        Returns a Sample object from the database.
        If the requested sample doesn't exist, it'll get generated via the INTERFACE_MODULE.generate_sample() function.
        This causes the function call to block until it's done. (possibly for hours)

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
        pickle_name = self._db_dict[params_hashed]['pickle_name']
        # load the Sample from disk
        print("\tRetrieved sample ", params_hashed, " from db, loading its representation from ", pickle_name, ".", sep="'")
        extracted_sample = self._unpickle_sample(pickle_name)
        # Do a sanity check of the parameters, just in case of a hash collision
        if not complete_rosparams == extracted_sample.parameters:
            raise LookupError("Got a sample with hash " + params_hashed + ", but its parameters didn't match the requested parameters. (Hash function collision?)", complete_rosparams, extracted_sample.parameters)
        return extracted_sample

    def _pickle_sample(self, sample, pickle_name, override_existing=False):
        """
        Helper method for saving samples to disk.

        :param sample: The Sample which should be pickled.
        :param pickle_name: The name of the Sample's pickled representation.
        """
        pickle_path = os.path.abspath(os.path.join(self._sample_dir_path, pickle_name))
        # Safety check, don't just overwrite other pickles!
        if not override_existing and os.path.exists(pickle_path):
            raise ValueError("A pickle file already exists at the calculated location:", pickle_path)
        print("\tPickling Sample object for later usage to:", pickle_path)
        with open(pickle_path, 'wb') as sample_pickle_handle:
            pickle.dump(sample, sample_pickle_handle)

    def _unpickle_sample(self, pickle_name):
        """
        Helper method for loading samples from disk

        :param pickle_name: Path to the Sample's pickled representation.
        """
        pickle_path = os.path.abspath(os.path.join(self._sample_dir_path, pickle_name))
        with open(pickle_path, 'rb') as sample_pickle_handle:
            sample = pickle.load(sample_pickle_handle)
            if not isinstance(sample, Sample):
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
