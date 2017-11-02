#!/usr/bin/env python3
"""
Classes for different ways to measure the map matcher's performance, based on information in a sample.
Only classes that end with Measure (not *Function classes) take a Sample as __call__ parameter and are meant for immediate usage with evaluation_function.py.
"""

import numpy as np
import matplotlib.pyplot as plt

def rotation_to_translation_error(rotation_error, submap_size):
    """
    Turns a rotation error into a translation error by looking at the upper bound of translation errors that
    are caused by that rotation error.
    For this, the submap size is used, which should be the distance to the submap's point farthest from the origin.
    Returns a list of translation errors if a list of rotation errors were given.
    Otherwise, a single value will be returned. (behaves just like other numpy functions)
    :params rotation_error: The rotation error or list of rotation errors in degrees.
    :params submap_size: The submap's size in the same unit as your translation errors.
    """
    rotation_error_rad = np.deg2rad(rotation_error)
    return 2*submap_size * np.sin(rotation_error_rad/2)

class PerformanceMeasure(object):
    """
    Contains some common methods used in all PerformanceMeasures.
    Also contains a utility class function to create a PerformanceMeasure subclass from a dict.

    Other than that, it mainly exists for showing what is necessary to implement when adding a new PerformanceMeasure subclass.
    And to allow isinstance tests with the PerformanceMeasure type.
    """

    AVAILABLE_TYPES = ['LogisticTranslationErrorMeasure', 'LogisticMaximumErrorMeasure', 'MixerMeasure', 'NrMatchesMeasure']

    def __init__(self):
        """
        Placeholder constructor, will maybe do sth in the future...
        """
        pass

    def __str__(self):
        """
        Placeholder str representation, meant for LaTeX mathmode parser in matplotlib.
        """
        return u"$Performance Measure$"

    def __call__(self, sample):
        """
        Placeholder call method, so one could directly call an PerformanceMeasure object with a given sample.
        """
        raise RuntimeError("Shouldn't call (or instantiate...) the PerformanceMeasure superclass")
        return sample.something * 1337 # Do some magic with the sample's data

    def _prepare_plot(self, x_min, x_max, resolution=1000):
        x_space = np.linspace(x_min, x_max, resolution)
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel(str(self))
        ax.plot(x_space, self(x_space), label=str(self))
        ax.set_xlim(x_min, x_max)
        return fig, ax

    def plot(self, path, x_min, x_max):
        """
        Plots the Performance with continuous input values from x_min to x_max.
        """
        fig, ax = self._prepare_plot(x_min, x_max)
        fig.savefig(path)
        fig.clf()

    @classmethod
    def from_dict(cls, measure_dict):
        if measure_dict['type'] == cls.AVAILABLE_TYPES[0]: # LogisticTranslationErrorMeasure
            return LogisticTranslationErrorMeasure(max_relevant_error=measure_dict['max_relevant_error'])
        elif measure_dict['type'] == cls.AVAILABLE_TYPES[1]: # LogisticMaximumErrorMeasure
            return LogisticMaximumErrorMeasure(submap_size=measure_dict['submap_size'], max_relevant_error=measure_dict['max_relevant_error'])
        elif measure_dict['type'] == cls.AVAILABLE_TYPES[2]: # MixerMeasure
            measure_a = PerformanceMeasure.from_dict(measure_dict['measure_a'])
            measure_b = PerformanceMeasure.from_dict(measure_dict['measure_b'])
            return MixerMeasure(measure_a, measure_b, measure_dict['weight_b'])
        elif measure_dict['type'] == cls.AVAILABLE_TYPES[3]: # NrMatchesMeasure
            return NrMatchesMeasure(measure_dict['expected_nr_matches'])
        else:
            raise ValueError("Type not available", cls.AVAILABLE_TYPES)

class LogisticFunction(PerformanceMeasure):
    """
    PerformanceMeasure that uses the logistic function in its raw form:
    f(x) = 1 - l / (1 + e^(-k(x-x0)))
    with x being float.
    """
    def __init__(self, l=1, x0=0, k=1):
        """
        Initialize with setting parameters
        :param l: Maximum the function approaches for x towards infinity.
        :param x0: The point where the curve passes f(x)=0.5, and where it has a saddle point.
        :param k: Scales curve width.
        """
        super().__init__()
        self.l = float(l)
        self.x0 = float(x0)
        self.k = float(k)

    def __str__(self):
        return u"$1 - \\frac{" + str(self.l) + u"}{1 + e^{-" + str(round(self.k, 3)) + u" * (x - " + str(self.x0) + u")}}$"

    def __call__(self, x):
        return 1 - self.l / (1 + np.exp(-self.k * (x - self.x0)))

class LogisticTranslationErrorMeasure(LogisticFunction):
    """
    PerformanceMeasure that uses the logistic function to map possible translation errors between 0 and 1.
    High translation errors will be mapped close to 0, low ones close to 1.
    """
    def __init__(self, max_relevant_error, min_relevant_error=0):
        """
        Initialize with setting parameters
        :param max_relevant_error: The maximum translation error that should still be distinguishable from higher errors. (i.e. mapped not too close to 0)
        :param min_relevant_error: Same for the minimum, defaults to 0.
        """
        self.x_min_y_value = 0.98 # determines how close to 1 x_min gets mapped (x_max gets mapped to 1-x_min_y_value)
        self.min_relevant_error = float(min_relevant_error)
        self.max_relevant_error = float(max_relevant_error)
        x0 = (self.max_relevant_error - self.min_relevant_error) / 2
        # Find k by using point (max_relevant_error, x_min_y_value)
        # Solved logistic function for k:
        k = np.log(1 / ((1 / self.x_min_y_value) - 1) / x0 - self.max_relevant_error)
        # min_relevant_error will fit, because of the function's symmetry and the given x0
        super().__init__(1, x0, k)

    def __call__(self, sample):
        if isinstance(sample, np.ndarray) or isinstance(sample, float): # Special case for plotting the function
            return super(LogisticTranslationErrorMeasure, self).__call__(sample)

        # Put each translation error through the Logistic function (super().__call__)
        match_errors = [super(LogisticTranslationErrorMeasure, self).__call__(err_t) for err_t in sample.translation_errors]
        # Sum them up and normalize with the number of matches
        if not sample.nr_matches == 0:
            return sum(match_errors, 0) / sample.nr_matches
        else:
            return 0

    def plot(self, path, x_min, x_max):
        fig, ax = self._prepare_plot(x_min, x_max)
        ax.set_xlabel("x: Translation Errors")
        ax.scatter([self.min_relevant_error, self.max_relevant_error],
                   [self(self.min_relevant_error), self(self.max_relevant_error)],
                   c='r', label=u"${(" + str(round(self.min_relevant_error, 2)) + u"," + str(round(self(self.min_relevant_error), 2)) + "),(" + str(round(self.max_relevant_error, 2)) + u"," + str(round(self(self.max_relevant_error), 2)) + u")}$")
        ax.legend(loc='lower left')
        ax.set_ylim(0,1)
        fig.savefig(path)
        fig.clf()

class MixerMeasure(PerformanceMeasure):
    """
    Mixes two other measures.
    """

    def __init__(self, measure_a, measure_b, weight_b):
        """
        Initialize with setting parameters
        :param measure_a: Performance measure a
        :param measure_b: Performance measure b
        :param weight_b: The weight of measure b. The measure_a's weight will be 1-weight_b.
        """
        super().__init__()
        self.measure_a = measure_a
        self.measure_b = measure_b
        self.weight_b = weight_b

    def __call__(self, sample):
        return (1 - self.weight_b) * self.measure_a(sample) + self.weight_b * self.measure_b(sample)

class LogisticMaximumErrorMeasure(LogisticTranslationErrorMeasure):
    """
    PerformanceMeasure that uses the logistic function to map possible errors between 0 and 1.
    High errors will be mapped close to 0, low ones close to 1.
    This PerformanceMeasure considers both translation and rotation errors.
    Rotation errors will be turned into translation errors:
        By considering the maximum translation error that could've been caused by the rotation
        around the submap's origin with a given distance to the point farthest away from the origin. (submap size)
    """
    def __init__(self, submap_size, max_relevant_error, min_relevant_error=0):
        """
        Initialize with setting parameters
        :param submap_size: The size of each submap, used for turning rotation errors into translation errors.
        :param max_relevant_error: The maximum translation error that should still be distinguishable from higher errors. (i.e. mapped not too close to 0)
        :param min_relevant_error: Same for the minimum, defaults to 0.
        """
        self.submap_size = submap_size
        super().__init__(max_relevant_error, min_relevant_error)

    def __call__(self, sample):
        if isinstance(sample, np.ndarray) or isinstance(sample, float): # Special case for plotting the function
            return super(LogisticMaximumErrorMeasure, self).__call__(sample)

        # Iterate over rotation and translation error lists simultaneously and chose the biggest errors
        considered_errors = [max(err_t, err_r) for err_t, err_r in
                             zip(sample.translation_errors,
                                 rotation_to_translation_error(sample.rotation_errors, self.submap_size))]
        match_errors = [super(LogisticMaximumErrorMeasure, self).__call__(err) for err in considered_errors]
        # Sum them up and normalize with the number of matches
        if not sample.nr_matches == 0:
            return sum(match_errors, 0) / sample.nr_matches
        else:
            return 0

class NrMatchesMeasure(PerformanceMeasure):
    """
    Uses the function x / (a + x) to map the number of matches to [0,1).
    More matches means this measure gets closer to 1.
    """
    def __init__(self, expected_nr_matches):
        """
        :param expected_nr_matches: A number of matches which is already considered quite good for the dataset
                                    Will be used to calculate a, so that the function passes point (a, 0.98)
        """
        self.expected_nr_matches = int(expected_nr_matches)
        # Can be used to tune how much space the measure has left for matches above the expected_nr_matches threshold
        self.expected_nr_matches_y_value = 0.75
        self.a = self.expected_nr_matches * (1 - self.expected_nr_matches_y_value) / self.expected_nr_matches_y_value

    def __str__(self):
        return u"$\\frac{x}{" + str(round(self.a, 3)) + u" + x}$"

    def __call__(self, sample):
        if isinstance(sample, np.ndarray) or isinstance(sample, int): # Special case for plotting the function
            return sample / (self.a + sample)

        return sample.nr_matches / (self.a + sample.nr_matches)

    def plot(self, path, x_min, x_max):
        fig, ax = self._prepare_plot(x_min, x_max)
        ax.set_xlabel("x: Number of Matches")
        ax.scatter([self.expected_nr_matches],
                   [self(self.expected_nr_matches)],
                   c='r', label=u"${(" + str(round(self.expected_nr_matches, 2)) + u"," + str(round(self(self.expected_nr_matches), 2)) + ")}$")
        ax.legend(loc='lower left')
        ax.set_ylim(0,1)
        fig.savefig(path)
        fig.clf()

class SinusTestFunction(PerformanceMeasure):
    """
    PerformanceMeasure that can be used for testing without any samples.
    It simply models sin(x)=a*x*sin(b*x), with x being a float.
    """
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __str__(self):
        return u"$" + str(self.a) + u"x * \sin(" + str(self.b) + u"x)$"

    def __call__(self, x):
        return self.a * x * np.sin(self.b * x)

if __name__ == '__main__': # don't execute when module is imported
    import argparse # for the cmd-line interface

    def command_line_interface():
        parser = argparse.ArgumentParser(description="tests performance measures")
        parser.add_argument('type', help="Type of plot to create")
        parser.add_argument('path', help="Path for plot")
        parser.add_argument('--min', help="x_min for plot range", required=True, type=float)
        parser.add_argument('--max', help="x_max for plot range", required=True, type=float)
        parser.add_argument('params', help="list of params, depends on chosen type", nargs='*')
        args = parser.parse_args()
        
        if args.type == 'logistic_raw':
            print("Logistic measure with x0 =", args.params[0], "and k =", args.params[1])
            l = LogisticFunction(x0 = args.params[0], k=args.params[1])
            l.plot(args.path, args.min, args.max)
        elif args.type == 'logistic_tra':
            print("Logistic translation measure with max_relevant_error =", args.params[0])
            l = LogisticTranslationErrorMeasure(max_relevant_error = args.params[0])
            l.plot(args.path, args.min, args.max)
        elif args.type == 'nr_matches':
            print("Measure for nr matches with expected nr matches =", args.params[0])
            l = NrMatchesMeasure(args.params[0])
            l.plot(args.path, args.min, args.max)
        else:
            print("unknown type")

    command_line_interface()
