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


    def __init__(self, sample_db, default_rosparams, _optimized_rosparams_definition):
        """
        Creates an EvaluationFunction object.
        
        :param sample_db: A sample database object.
        :param default_rosparams: A dict that contains all rosparams required for running the map matcher
                                  with the default values.
        :param optimized_rosparams_definition: A dict that contains the dict of rosparams which are beeing optimized in this experiment.
                                               Expects a dict that contains one or multiple entires, each containing rosparam_name, min_bound and max_bound.
        """
        self.sample_db = sample_db
        self._default_rosparams = default_rosparams
        self._optimized_rosparams_definition = _optimized_rosparams_definition

    def evaluate(self, optimized_rosparams):
        """
        Evaluates the function at X=rosparams and returns y=metric(map_matcher_result(X)).
        rosparams is the default_rosparams dict updated with the optimized_rosparams dict.
        
        :param optimized_rosparams: 
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
