"""
Here we take three expansions of a unit cell into a supercell and use these to 
try and predict the limits of the expansion beyond that we can compute in a 
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
from collections import OrderedDict, Counter

import numpy as np
import networkx as nx

from scipy.optimize import curve_fit


class PersistenceLimiter():
    def __init__(self, exp_3, exp_5):
        self.function_limit = 10000
        self.exp_3 = exp_3
        self.exp_5 = exp_5
    
    
    def gen_inf_freq(self, exp_3, exp_5):
        exp_inf = Counter()
        xs = np.array([3, 5])

        for point, frequency in exp_3.items():
            ys = [frequency]
            ys.append(exp_5[point])
            limit = self.find_limit(xs, ys)
            exp_inf[point] = limit


    def find_limit(self, xs, ys):
        func = self.get_base_func(xs, ys)
        popt, pcov = curve_fit(func, xs, ys)

        return func(self.function_limit, *popt)


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