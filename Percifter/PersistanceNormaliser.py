"""
Author: Cameron Hargreaves

This simple class takes in a list of persistence points and "normalises" these
so that the total sum of points is equal to one

We include a method for the modified bottleneck distance on these points
"""
import os
import pickle as pk

from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy import ndimage

def main():
    test_string = '/home/cameron/Documents/tmp/PersGen/icsd_000373/icsd_000373_anions.pers'
    pers_points = pk.load(open(test_string, "rb"))
    x = PersistenceNorm(pers_points)
    x._count_points()
    x.normalise_points()
    x.maximum_matching(deepcopy(x.norm_list), deepcopy(x.norm_list))
    print()

class PersistenceNorm():
    def __init__(self, points):
        self.points = points
        self.counter_list = []
        self._count_points()

    def _count_points(self, dp=5):
        """
        Save a list of Counters to self in the form:
        self.point_counter = [Counter(H_0 points), Counter(H_1 points), ...]

        Parameters:
        dp: Int, default 5. The number of decimal places to round to
        """
        counter_list = []

        points = self.points
        # In case it has already been reduced
        if type(points) is Counter:
            self.point_counter = points

        # Standard output should be a list of 2D numpy arrays
        elif type(points) is list:
            for i, diagram in enumerate(points):
                points[i] = diagram[~np.isinf(diagram).any(axis=1)]

            for homology_group in points:
                # Round each of these to 5dp and then cast to a list of tuples
                # so that can apply Counter
                homology_group = [tuple(x) for x in np.round(homology_group, dp)]
                count = Counter(homology_group)
                counter_list.append(count)

        self.counter_list = counter_list

    def normalise_points(self, counter_list=None):
        if counter_list == None:
            if self.counter_list == None:
                self._count_points()
                counter_list = self.counter_list
            else:
                counter_list = self.counter_list

        norm_list = []

        for i, point_counter in enumerate(counter_list):
            total = sum(point_counter.values(), 0.0)
            for key in point_counter:
                point_counter[key] /= total
            norm_list.append(point_counter)

        self.norm_list = norm_list

    def maximum_matching(self, freq_self, other):
        """
        Perform a bartitite maximal matching of two frequency counts,
        recursively called until all points are matched together
        """
        if freq_self == None:
            freq_self = self.norm_list

        for i, group in enumerate(freq_self):
            other_group = other[i+1]
            group, other_group = self.remove_matching_pairs(group, other_group)
            self._bipartite_match(group, other_group)

    def remove_matching_pairs(self, freq, other):
        # First remove all points that match perfectly from both lists
        pop_list = []
        for point in freq.keys():
            if point in other.keys() and freq[point] == other[point]:
                pop_list.append(point)

        for point in pop_list:
            freq.pop(point)
            other.pop(point)

        return freq, other

    def _bipartite_match(self, freq, other):
        """
        TODO: Very inefficient implementation, optimise with Hopcroft-Karp algo
        """
        if len(freq) == 0:
            return
        elif len(other) == 0:
            return

        x, x_counts = zip(*freq.items())
        y, y_counts = zip(*other.items())

        dist_matrix = cdist(x[1:], y[1:])
        (min_x, min_y) = ndimage.minimum_position(dist_matrix)

        self.match(x[min_x], y[min_y])

        print()

    def match(self, x, y):
        print()


if __name__ == "__main__":
    main()
