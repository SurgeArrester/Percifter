"""
Author: Cameron Hargreaves

This simple class takes in a list of persistence points and "normalises" these
so that the total sum of points is equal to one

We include a method for the modified bottleneck distance on these points that
carries forward a recursive bipartite matching on the two, returning the shared
counts and the distance between these, and uses this as a modified score metric

The returned value can be used as a distance metric between each homology group

"""

import os
import pickle as pk

from collections import Counter, OrderedDict
from copy import deepcopy

import numpy as np

from scipy.spatial.distance import cdist, squareform, euclidean
from scipy.optimize import linear_sum_assignment

from ortools.graph import pywrapgraph

def main():
    test_string1 = '/home/cameron/Dropbox/University/PhD/Percifter/Percifter/Percifter/OutFiles/Li01.pers'
    test_string2 = '/home/cameron/Dropbox/University/PhD/Percifter/Percifter/Percifter/OutFiles/Li02.pers'

    pers_points = pk.load(open(test_string1, "rb"))
    x = PersistenceNorm(pers_points)
    pers_points = pk.load(open(test_string2, "rb"))
    y = PersistenceNorm(pers_points)

    scores = x.flow_norm_bottleneck(y)
    print(f"\n{scores}")

    # score = x.normalised_bottleneck(y)
    # print(score)

class PersistenceNorm():
    FP_MULTIPLIER = 100000

    def __init__(self, points, verbose=True):
        self.points = points
        self.verbose = verbose
        self.counter_list = []
        self._count_points()
        self._normalise_points()

    def flow_norm_bottleneck(self, comp2, comp1=None):
        """
        Use the minimal cost multicomodity flow algorithm to generate a distance
        metric between two ratio disctionaries
        """
        if comp1 == None:
            comp1 = deepcopy(self.norm_list)

        if type(comp2) == PersistenceNorm:
            comp2 = deepcopy(comp2.norm_list)

        scores = []

        for hom_group_1, hom_group_2, i in zip(comp1, comp2, range(len(comp1))):
            print(f"\nHomology Group {i}")
            scores.append(self._flow_dist(hom_group_1, hom_group_2))

        return scores

    def _flow_dist(self, hom_group_1, hom_group_2):
        start_nodes, end_nodes, labels, capacities, costs, supplies = self._generate_parameters(hom_group_1, hom_group_2)

        # Google OR-tools only take integer values, so we multiply our floats
        # by self.FP_MULTIPLIER and cast to int
        capacities = [int(x * self.FP_MULTIPLIER) for x in capacities]
        supplies = [int(x * self.FP_MULTIPLIER) for x in supplies]
        costs = [int(x * self.FP_MULTIPLIER) for x in costs]

        # Due to rounding errors, the two supplies may no longer be equal to one
        # another. We add the difference to the largest value in the smaller set
        # to allow this to be processed and minimise the error
        source_tot = sum([x for x in supplies if x > 0])
        sink_tot = -sum([x for x in supplies if x < 0])

        while sink_tot < source_tot:
            supplies[supplies.index(min(supplies))] -= 1
            sink_tot = -sum([x for x in supplies if x < 0])

        while source_tot < sink_tot:
            supplies[supplies.index(max(supplies))] += 1
            source_tot = sum([x for x in supplies if x > 0])

        # Instantiate a SimpleMinCostFlow solver
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()

        # Add each arc.
        for i in range(0, len(start_nodes)):
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                        capacities[i], costs[i])

        # Add node supplies.
        for i in range(0, len(supplies)):
            min_cost_flow.SetNodeSupply(i, supplies[i])

        feasibility_status = min_cost_flow.Solve()

        if feasibility_status == min_cost_flow.OPTIMAL:
            dist = min_cost_flow.OptimalCost() / self.FP_MULTIPLIER ** 2

            if self.verbose:
                print('Distance Score:', min_cost_flow.OptimalCost() / self.FP_MULTIPLIER ** 2)
                print('Arc  \t|\t  Flow / Capacity  \t|\t Cost')
                for i in range(min_cost_flow.NumArcs()):
                    cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                    print('%s -> %s \t | \t %3.5s / %3.5s \t|\t %3.5s' % (
                        #min_cost_flow.Tail(i),
                        labels[min_cost_flow.Tail(i)].split('_')[0],
                        #min_cost_flow.Head(i),
                        labels[min_cost_flow.Head(i)].split('_')[0],
                        min_cost_flow.Flow(i) / self.FP_MULTIPLIER,
                        min_cost_flow.Capacity(i) / self.FP_MULTIPLIER,
                        cost / self.FP_MULTIPLIER ** 2 ))

            return dist

        else:
            return "Infeasible solution"

    def _generate_parameters(self, source, sink):
        start_nodes = []
        start_labels = []
        end_nodes = []
        end_labels = []

        capacities = []
        costs = []
        supply_tracker = OrderedDict()

        for i, key_value_source in enumerate(source.items()):
            for j, key_value_sink in enumerate(sink.items()):
                start_nodes.append(i)
                start_labels.append(key_value_source[0])
                end_nodes.append(j + len(source))
                end_labels.append(key_value_sink[0])
                capacities.append(min(key_value_source[1], key_value_sink[1]))
                costs.append(euclidean(key_value_sink[0], key_value_source[0]))

        for lab in start_labels:
            supply_tracker[str(lab) + "_source"] = source[lab]

        for lab in end_labels:
            supply_tracker[str(lab) + "_sink"] = -sink[lab]

        labels = list(supply_tracker.keys())
        supplies = list(supply_tracker.values())

        return start_nodes, end_nodes, labels, capacities, costs, supplies

    def normalised_bottleneck(self, other, freq_self=None):
        """
        Perform a bartitite maximal matching of two frequency counts,
        recursively called until all points are matched together
        This only takes into account the homology groups of self and will not
        match with higher dimensions in other and currently will break if self
        has more dimensions than other
        """
        if freq_self == None:
            freq_self = deepcopy(self.norm_list)

        if type(other) == PersistenceNorm:
            other = deepcopy(other.norm_list)

        scores = []
        for i, group in enumerate(freq_self):
            matched_pairs = []
            other_group = other[i]

            # Recursively match the closest points in each group, calc their
            # distance and shared cardinality, and append to matched_pairs
            matching = self._bipartite_match(group, other_group, matched_pairs)

            # For each of these points sum the product of their distance and
            # shared cardinality
            scores.append(sum(x[2] * x[3] for x in matching))

        return scores

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
            counter_list = points

        # Standard output should be a list of 2D numpy arrays
        elif type(points) is list:
            # Strip the infinite points
            for i, diagram in enumerate(points):
                points[i] = diagram[~np.isinf(diagram).any(axis=1)]

            # Round each of these to 5dp, cast to a list of tuples
            # and apply a Counter
            for homology_group in points:
                homology_group = [tuple(x) for x in np.round(homology_group,dp)]
                count = Counter(homology_group)
                counter_list.append(count)

        self.counter_list = counter_list

    def _normalise_points(self, counter_list=None):
        """Performs standard normalisation in each dimension to give ratios"""
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

    def _bipartite_match(self, freq, other, matched_pairs):
        """
        Possibly inefficient implementation to run max bipartite matching
        TODO: Optimise with Hopcroft-Karp algorithm?
        """
        # If we have reached the base case simply add the remaining points
        # unmatched
        if len(freq) == 0:
            for point in other.keys():
                matched_pairs.append(('_', point, np.linalg.norm(point),
                                                  other[point]))
            return matched_pairs

        elif len(other) == 0:
            for point in freq.keys():
                matched_pairs.append((point, '_', np.linalg.norm(point),
                                                  freq[point]))
            return matched_pairs

        # Unpack the dictionaries into lists of points
        x = tuple(freq.keys())
        y = tuple(other.keys())

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
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i),other[y_i]))
                freq[x_i] -= other[y_i]
                other.pop(y_i)

            elif freq[x_i] < other[y_i]:
                matched_pairs.append((x_i, y_i, euclidean(x_i, y_i), freq[x_i]))
                other[y_i] -= freq[x_i]
                freq.pop(x_i)

        # After we have reduced the dictionaries sizes, call this again on the
        # remaining values
        self._bipartite_match(freq, other, matched_pairs)

        return matched_pairs

if __name__ == "__main__":
    main()
