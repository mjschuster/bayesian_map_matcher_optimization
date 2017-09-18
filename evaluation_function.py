"""
Contains the code for modelling the EvaluationFunction via discrete samples.
"""

class EvaluationFunction(object):
    """
    Models the evaluation function that is optimized by the gaussian process.

    In our use-case, feature-based map matcher evaluation, the evaluation function can't be defined analytically.
    Instead, this class models it as a set of sample points for which the map matcher pipeline was evaluated.

    The class uses the _sample_db member to store known samples.
    This should reduce the time subsequent experiments will require, after a bunch of samples have already been generated.
    """

    METRICS = ['mean_translation_error']

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

        self.sample_db = sample_db
        self._default_rosparams = default_rosparams
        self._optimization_definitions = optimization_definitions

    def evaluate(self, args): # TODO howto args.. / **args ?
        """
        Evaluates the function at the given parameters.
        This method supplies the interface the Optimizer needs and checks whether the arguments are bounded correctly.
        """
        optimized_rosparams = {}
        return self.get_metric_at(optimized_rosparams)

    def get_metric_at(self, optimized_rosparams):
        """
        Calculates and returns the current metric at X and returns it, where X is the default_rosparams
        dict updated with the given optimized_rosparams dict.
        
        :param optimized_rosparams: dict of rosparams with values in the defined bounds.
        """

        # Error handling
        for param in self._optimized_rosparams.keys():
            if not param in optimized_rosparams:
                raise ValueError(str(param) + " should get optimized, but wasn't in given dict of optimized parameters.")
        for param in optimized_rosparams.keys():
            if not param in self._optimized_rosparams:
                raise ValueError(str(param) + " shouldn't get optimized, but was in given dict of optimized parameters.")
        print("Evaluating sample at", optimized_rosparams.items())
        # Create the full set of parameters by updating the default parameters with the optimized parameters.
        rosparams = self._default_rosparams.copy()
        rosparams.update(optimized_rosparams)

        sample = self._sample_db.get_sample(rosparams)
        print("Got sample", sample)

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
        * origin: A string which describes where that sample was generated. Probably should be a path to the map matcher results directory.
    """

    def __init__(self):
        """
        Creates a function sample object with empty data contents.
        Fill with available data by accessing its propteries.
        """
        self.translation_errors = []
        self.rotation_errors = []
        self.parameters = {}
        self.origin = None
