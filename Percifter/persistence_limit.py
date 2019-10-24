"""
Here we take two expansions of a unit cell into a supercell and use these to
predict the limits of the expansion beyond what we can compute in a
reasonable time

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

"""
from collections import Counter

import numpy as np
import networkx as nx

from scipy.optimize import curve_fit

class PersistenceLimits():
    """
    Take two counters of persistence point frequencies and by using non-linear
    least squares to fit a known function to a curve

    Params
    exp_3, exp_5: Counters for two expansions of a supercell
    """
    def __init__(self, exp_5, exp_7, exp_9):
        self.function_limit = 1000

        self.exp_5 = exp_5
        self.exp_7 = exp_7
        self.exp_9 = exp_9

        exp_inf = self.gen_inf_freq()
        self.exp_inf = self.remove_error(exp_inf)

    def remove_error(self, exp_inf):
        '''
        The sum of all the ratios should sum to 1. Unfortunately there are
        unavoidable errors in precision which is compounded when taking limits
        up to infinity. Here we use a clumsy method to remove this error and
        also remove all points with zero frequency

        TODO: Make this more elegant
        '''
        ratio_sum = sum([x for x in list(exp_inf.values())])

        if ratio_sum == 1:
            return exp_inf

        i = 1
        while round(ratio_sum, 9) != 1:
            epsilon = (1 - ratio_sum) / len(exp_inf)
            for key in list(exp_inf.keys()):
                exp_inf[key] = exp_inf[key] + epsilon

                if (round(exp_inf[key], 8) == 0) or (exp_inf[key] < 0):
                    exp_inf.pop(key)

            ratio_sum = sum([x for x in list(exp_inf.values())])
            i += 1
            print(ratio_sum)
        return exp_inf


    def gen_inf_freq(self):
        '''
        Use linear least squares to create a new counter for the infinite
        expansion. In each of our defined functions, the limit is given by the
        value of c in each of theese, which is the last value of the optimal
        parameter array.
        '''
        exp_inf = Counter()
        xs = np.array([5, 7, 9])

        for point, frequency in self.exp_5.items():
            ys = [frequency]
            ys.append(self.exp_7[point])
            ys.append(self.exp_9[point])

            limit = self.find_limit(xs, ys)
            exp_inf[point] = limit

        return exp_inf

    def find_limit(self, xs, ys):
        func = self.get_base_func(xs, ys)
        popt, pcov = curve_fit(func, xs, ys, bounds=(0, 1))

        return popt[-1]
        # return (func(self.function_limit, *popt), func)

    def get_base_func(self, xs, ys):
        if ys[0] < ys[1]:
            func = self._inv_recip_func

        elif ys[0] > ys[1]:
            func = self._recip_func

        elif ys[0] == [ys[1]]:
            func = self._const_func

        return func

    def _recip_func(self, x, a, b, c):
        return (a / (x - b)) + c

    def _inv_recip_func(self, x, a, b, c):
        return (-a / (x - b)) + c

    def _const_func(self, x, c):
        return c
