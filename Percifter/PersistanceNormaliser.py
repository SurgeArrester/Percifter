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

from scipy import ndimage
from scipy.spatial.distance import cdist, squareform, euclidean
from scipy.optimize import linear_sum_assignment

def main():
    test_string1 = '/home/cameron/Documents/tmp/PersGen/icsd_000373/icsd_000373_anions.pers'
    test_string2 = '/home/cameron/Documents/tmp/PersGen/icsd_001017/icsd_001017_anions.pers'
    pers_points = pk.load(open(test_string, "rb"))
    x = PersistenceNorm(pers_points)
    x._count_points()
    x.normalise_points()
    x.normalised_bottleneck(deepcopy(x.norm_list), deepcopy(x.norm_list))
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
            # Strip the infinite points
            for i, diagram in enumerate(points):
                points[i] = diagram[~np.isinf(diagram).any(axis=1)]

            # Round each of these to 5dp and then cast to a list of tuples
            # and apply a Counter
            for homology_group in points:
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

    def normalised_bottleneck(self, freq_self, other):
        """
        Perform a bartitite maximal matching of two frequency counts,
        recursively called until all points are matched together
        This only takes into account the homology groups of self and will not
        match with higher dimensions in other and currently will break if self
        has more dimensions than other
        """
        if freq_self == None:
            freq_self = self.norm_list

        scores = []
        for i, group in enumerate(freq_self):
            matched_pairs = []
            other_group = other[i]

            # Take out the identical points as these sum to zero
            group, other_group = self.remove_matching_pairs(group, other_group, matched_pairs)
            matching = self._bipartite_match(group, other_group, matched_pairs)

            # For each of these points sum the product of their distance and
            # matching frequency
            scores.append(sum(x[2] * x[3] for x in matching))

        return scores

    def remove_matching_pairs(self, freq, other, matched_pairs):
        # First remove all points that match perfectly from both lists
        pop_list = []
        for point in freq.keys():
            if point in other.keys() and freq[point] == other[point]:
                pop_list.append(point)
                matched_pairs.append((point, point, 0, freq[point]))

        for point in pop_list:
            freq.pop(point)
            other.pop(point)

        return freq, other

    def _bipartite_match(self, freq, other, matched_pairs):
        """
        Very inefficient implementation to run max bipartite matching
        TODO: Optimise with Hopcroft-Karp algo
        """
        # If we have reached the base case simply add the remaining points
        # unmatched
        if len(freq) == 0:
            for point in other.keys():
                matched_pairs.append(('_', point, np.linalg.norm(point), other[point]))
            return matched_pairs

        elif len(other) == 0:
            for point in freq.keys():
                matched_pairs.append((point, '_', np.linalg.norm(point), freq[point]))
            return matched_pairs

        # Unpack and zip the dictionaries into lists of keys and values
        x, x_counts = zip(*freq.items())
        y, y_counts = zip(*other.items())

        # Compute a distance matrix of these and then use the hungarian
        # algorithm in linear_sum_assignment() to create the minimal bipartite
        # matching of these two
        dist_matrix = cdist(x, y)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        x_matched = [x[i] for i in row_ind]
        y_matched = [y[i] for i in col_ind]

        for (x_i, y_i) in zip(x_matched, y_matched):
            if freq[x_i] == other[y_i]:
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i), freq[x_i]))
                freq.pop(x_i)
                other.pop(y_i)

            elif freq[x_i] > other[y_i]:
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i), other[y_i]))
                freq[x_i] -= other[y_i]
                other.pop(y_i)

            elif freq[x_i] < other[y_i]:
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i), freq[x_i]))
                other[y_i] -= freq[x_i]
                freq.pop(x_i)

        self._bipartite_match(freq, other, matched_pairs)

        return matched_pairs



if __name__ == "__main__":
    main()
