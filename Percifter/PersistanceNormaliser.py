"""
PersistenceNormaliser takes a list of persistence points, calculate their ratios
and provide methods for similarity metrics

Copyright (C) 2019  Cameron Hargreaves

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

--------------------------------------------------------------------------------

This simple class takes in a list of persistence points and "normalises" these
so that the total sum of points is equal to one, giving the ratio of each
persistence point

The input is two processed persistence files from CifToPers and the minimum
cost multi-commodity flow distance is calculated between the two using google
OR tools

We also include a deprecated method for the modified bottleneck distance on these
points that carries forward a recursive bipartite matching on the two, returning
the shared counts and the distance between these, and uses this as a modified
score metric

The returned value from flow_norm_bottleneck() can be used as a distance metric
between two persistence diagrams

LICENCE
OR-Tools

Copyright 2010 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
    test_string1 = './OutFiles/Li01.pers'
    test_string2 = './OutFiles/Li02.pers'

    pers_points = pk.load(open(test_string1, "rb"))
    x = PersistenceNorm(pers_points)
    pers_points = pk.load(open(test_string2, "rb"))
    y = PersistenceNorm(pers_points)

    scores = x.flow_norm_bottleneck(y)
    print(f"\n{scores}")


class PersistenceNorm():
    FP_MULTIPLIER = 100000000

    def __init__(self, points, verbose=True):
        self.points = points
        self.verbose = verbose
        self.counter_list = []
        self._count_points()       # Creates self.counter_list
        self._normalise_points()   # Creates self.norm_list

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

        for _, point_counter in enumerate(counter_list):
            total = sum(point_counter.values(), 0.0)
            for key in point_counter:
                point_counter[key] /= total
            norm_list.append(point_counter)

        self.norm_list = norm_list

    def flow_norm_bottleneck(self, comp2, comp1=None):
        """
        Use the minimal cost multi-commodity flow algorithm to generate a
        distance metric between two ratio dictionaries
        """
        if comp1 == None:
            comp1 = deepcopy(self.norm_list)

        if type(comp2) == PersistenceNorm:
            comp2 = deepcopy(comp2.norm_list)

        scores = []

        for hom_group_1, hom_group_2, i in zip(comp1, comp2, range(len(comp1))):
            if self.verbose:
                print(f"\nHomology Group {i}")
            scores.append(self._flow_dist(hom_group_1, hom_group_2))

        return scores

    def _flow_dist(self, hom_group_1, hom_group_2):
        """
        Use the minimal flow costing to find the distance between two
        persistence diagrams
        """
        start_nodes, end_nodes, labels, capacities, costs, supplies = \
            self._generate_parameters(hom_group_1, hom_group_2)

        # Google OR-tools only take integer values, so we multiply our floats
        # by self.FP_MULTIPLIER and cast to ints
        capacities = [int(x * self.FP_MULTIPLIER) for x in capacities]
        supplies = [int(x * self.FP_MULTIPLIER) for x in supplies]
        costs = [int(x * self.FP_MULTIPLIER) for x in costs]

        # Due to rounding errors, the two supplies may no longer be equal to one
        # another. We add the difference to the largest value in the smaller set
        # to make the sum of both capacities equal
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
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],
                                                        end_nodes[i],
                                                        capacities[i],
                                                        costs[i])

        # Add node supplies.
        for i in range(len(supplies)):
            min_cost_flow.SetNodeSupply(i, supplies[i])

        feasibility_status = min_cost_flow.Solve()

        if feasibility_status == min_cost_flow.OPTIMAL:
            dist = min_cost_flow.OptimalCost() / self.FP_MULTIPLIER ** 2

            if self.verbose:
                print('Arc \t\t\t\t\t   Flow \t  /Capacity  \t   Dist.  \t    Cost')
                for i in range(min_cost_flow.NumArcs()):
                    cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                    print('%-15s -> %-20s    %-15s/%-15s %-16s %-5s' % (
                        #min_cost_flow.Tail(i),
                        labels[min_cost_flow.Tail(i)].split('_')[0],
                        #min_cost_flow.Head(i),
                        labels[min_cost_flow.Head(i)].split('_')[0],
                        min_cost_flow.Flow(i) / self.FP_MULTIPLIER,
                        min_cost_flow.Capacity(i) /self.FP_MULTIPLIER,
                        costs[i] / self.FP_MULTIPLIER,
                        cost / self.FP_MULTIPLIER ** 2 ))

                print(f"Total Cost: {min_cost_flow.OptimalCost() / self.FP_MULTIPLIER**2}\n")
            return dist

        else:
            # If this has an infeasible solution (which shouldn't happen)
            return -1

    def _generate_parameters(self, source, sink):
        """
        Create the nodes, labels, costs, capacities and supply/demands for each
        node in the minimisation problem
        """
        start_nodes = []
        start_labels = []
        end_nodes = []
        end_labels = []

        capacities = []
        costs = []
        supply_tracker = OrderedDict()

        # Iterate over both lists to create labels for the directed graph
        for i, key_value_source in enumerate(source.items()):
            for j, key_value_sink in enumerate(sink.items()):
                start_nodes.append(i)
                start_labels.append(key_value_source[0])
                end_nodes.append(j + len(source))
                end_labels.append(key_value_sink[0])
                capacities.append(min(key_value_source[1], key_value_sink[1]))
                costs.append(euclidean(key_value_sink[0], key_value_source[0]))

        for label in start_labels:
            supply_tracker[str(label) + "_source"] = source[label]

        for label in end_labels:
            supply_tracker[str(label) + "_sink"] = -sink[label]

        labels = list(supply_tracker.keys())
        supplies = list(supply_tracker.values())

        return start_nodes, end_nodes, labels, capacities, costs, supplies

if __name__ == "__main__":
    main()
