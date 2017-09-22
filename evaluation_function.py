"""
Contains the code for modelling the EvaluationFunction via discrete samples.
"""

import numpy as np

class EvaluationFunction(object):
    """
    Models the evaluation function that is optimized by the gaussian process.

    In our use-case, feature-based map matcher evaluation, the evaluation function can't be defined analytically.
    Instead, this class models it as a set of sample points for which the map matcher pipeline was evaluated.

    The class uses the _sample_db member to store known samples.
    This should reduce the time subsequent experiments will require, after a bunch of samples have already been generated.
    """

    METRICS = ['test', 'mean_error']
    """
    Available metrics:
        * test: Metric for testing, will circumvent the sample generation and just model f(x)=50x*sin(50x).
                Will only work as long as just one parameter is optimized.
        * mean_translation_error: Will return mean translation error of a Sample's matches.
    """

    def __init__(self, sample_db, default_rosparams, optimization_definitions, used_metric):
        """
        Creates an EvaluationFunction object.
        
        :param sample_db: A sample database object.
        :param default_rosparams: A dict that contains all rosparams required for running the map matcher
                                  with the default values.
        :param optimization_definitions: A dict that contains the definitions which rosparams which are beeing optimized in this experiment and within which bounds.
                                         Expects a dict that contains one or multiple entires, each containing rosparam_name, min_bound and max_bound.
        :param used_metric: A string to tell which metric should be used. See EvaluationFunction.METRICS for all implemented metrics.
                            Some metrics may terminate the experiment, if your Sample instances don't contain the necessary data.
        """

        # error checking
        if not used_metric in EvaluationFunction.METRICS:
            raise ValueError("Unknown metric", used_metric, "possible values:", EvaluationFunction.METRICS)

        self._sample_db = sample_db
        self._default_rosparams = default_rosparams
        self._optimization_definitions = optimization_definitions
        self._used_metric = used_metric

    def evaluate(self, **optimized_rosparams):
        """
        This method supplies the interface for the Optimizer.
        Calculates and returns the current metric ("y") at the given parameters ("X").

        Will convert the optimized_rosparams (from the Optimizer-world) to the complete_rosparams which define 
        a Sample in the map matching pipeline.

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
            #if not type(p_value) == type(self._default_rosparams[p_name]):
            #    raise ValueError(str(p_name) + " changed its type to " + str(type(p_value)) + " during optimization." +\
            #                     " Type in default dict is " + str(type(self._default_rosparams[p_name])) + ".")

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

    def metric(self, sample):
        """
        Returns the active metric's value for a given Sample object.
        Its concrete behaviour depends on the used_metric parameter given to the init method.

        :param sample: An evaluation_function Sample.
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
            print("Sample's result with metric '" + self._used_metric + "': ", value)
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
