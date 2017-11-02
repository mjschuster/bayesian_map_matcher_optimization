#!/usr/bin/env python3
"""
Classes for different ways to measure the map matcher's performance, based on information in a sample.
Only classes that end with Measure (not *Function classes) take a Sample as __call__ parameter and are meant for immediate usage with evaluation_function.py.
"""

import numpy as np
import matplotlib.pyplot as plt

class PerformanceMeasure(object):
    """
    Contains some common methods used in all PerformanceMeasures.
    Also contains a utility class function to create a PerformanceMeasure subclass from a dict.

    Other than that, it mainly exists for showing what is necessary to implement when adding a new PerformanceMeasure subclass.
    And to allow isinstance tests with the PerformanceMeasure type.
    """

    AVAILABLE_TYPES = ['LogisticTranslationErrorMeasure']

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
    Therefor, the goal should be to maximize this measure.
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
        return sum(match_errors, 0) / sample.nr_matches

    def plot(self, path, x_min, x_max):
        fig, ax = self._prepare_plot(x_min, x_max)
        ax.set_xlabel("x: Translation Errors")
        ax.scatter([self.min_relevant_error, self.max_relevant_error],
                   [self(self.min_relevant_error), self(self.max_relevant_error)],
                   c='r', label=u"${(" + str(round(self.min_relevant_error, 2)) + u"," + str(round(self(self.min_relevant_error), 2)) + "),(" + str(round(self.max_relevant_error, 2)) + u"," + str(round(self(self.max_relevant_error), 2)) + u")}$")
        ax.legend(loc='lower left')
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
        else:
            print("unknown type")

    command_line_interface()
