'''
CifToPers takes a cif file and processes the persistence of this

Copyright (C) 2019 Cameron Hargreaves

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
--------------------------------------------------------------------------------

This class takes an input cif file, performs an expansion in x, y, z dimensions
and then returns a persistence diagram if we have minimal change on expansions
by writing to the given filepath

Parameters
cif_path : Path to search for cif file
output_path : Path to write final persistence diagram to

TODO
Replace diffpy with something more stable, appears to be returning incorrect
x, y, z coordinates.


'''
import os
import time
from collections import Counter

import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

import CifFile

import gudhi as gd

from copy import deepcopy

from diffpy.structure import loadStructure
from diffpy.structure.expansion.supercell_mod import supercell

from pymatgen.core import Structure
import sys
sys.path.append('/home/cameron/Dropbox/University/SecondYear/Percifter/diode/build/bindings/python')
sys.path.append('/home/cameron/Dropbox/University/SecondYear/Percifter/dionysus/build/bindings/python')

from diode import fill_alpha_shapes
from dionysus import Filtration, homology_persistence, init_diagrams

from sklearn.metrics.pairwise import euclidean_distances

from Niggli import Niggli
from PersistanceNormaliser import PersistenceNorm
from persistence_limit import PersistenceLimits

# Matplotlib throw a lot of annoying warnings!
import warnings
warnings.filterwarnings("ignore")

def main():
    input_path = '/home/cameron/Dropbox/University/SecondYear/Percifter/tests/GeorgTest/cif/'
    output_path = '/home/cameron/Dropbox/University/SecondYear/Percifter/tests/GeorgTest/pers/'

    # for file in os.listdir(output_path):
    #     if file[-4:] == "pers":
    #         pd = pk.load(open(output_path + file, "rb"))
    #         print(file)
    #         print(pd)
    #         print()
    
    for cif in os.listdir(input_path):
        x = CifToPers(input_path + cif, output_path + cif[:-4] + ".pers")

    # x = CifToPers(input_path, out_path + "icsd_000003.pers")
    # print(f"x is processed")
    # y = CifToPers(test_path, out_path + "icsd_000003_cations.pers")
    # print(f"y is processed")

    # scores = x.flow_dist(y)
    # print("The minimal flow score between each homology group is:")
    # print(f"H0: {scores[0]}\nH1: {scores[1]}\nH2: {scores[2]}")

class CifToPers():
    def __init__(self, input_path=None,
                       output_path=None,
                       simplicial_complex='alpha',
                       write_out=True,
                       DECIMAL_ROUNDING=2,
                       SIMILARITY_PERCENTAGE=0.05,
                       INITIAL_EXPANSION=[3, 3, 3],
                       MAX_EXPANSION=10):

        self.DECIMAL_ROUNDING = DECIMAL_ROUNDING
        self.SIMILARITY_PERCENTAGE = SIMILARITY_PERCENTAGE
        self.MAX_EXPANSION = MAX_EXPANSION
        self.INITIAL_EXPANSION = INITIAL_EXPANSION

        self.input_path = input_path
        self.complex = simplicial_complex


        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = input_path[:-4] + ".pers"

        try:
            self.cif = CifFile.ReadCif(input_path)
            self.structure = Structure.from_file(input_path)
            

        except Exception as e:
            print(f"Cannot read {input_path} make sure it is a correctly formatted cif file in utf8")
            print(e)

        namespace = list(self.cif.keys())[0]
        self.keys = list(self.cif[namespace].keys())

        self.lattice = self.generate_lattice(self.cif, namespace)

        self.xyz_coords = self.structure.cart_coords    
        self.unit_pers = self.generate_persistence(self.xyz_coords)
        self.threecell = deepcopy(self.structure) 
        self.threecell.make_supercell([3, 3, 3])
        self.threecell_coords = self.threecell.cart_coords
        self.threecell_coords[self.threecell_coords == 0.0] = 0.000000001
        self.gud_threecell_pers = self.generate_persistence(self.threecell_coords)

        self.threecell_pers = self.generate_persistence(self.threecell_coords)

        # simplices = fill_alpha_shapes(self.threecell_coords)
        # f = Filtration(simplices)
        # m = homology_persistence(f)
        # dgms = init_diagrams(m, f)
        # p_list = [Counter(), Counter(), Counter()]
        # for i, dgm in enumerate(dgms):
        #     for pt in dgm:
        #         c_list[i]
        #         print(i, pt.birth, pt.death)

        self.fivecell = deepcopy(self.structure) 
        self.fivecell.make_supercell([5, 5, 5])
        self.fivecell_coords = self.fivecell.cart_coords
        self.fivecell_coords[self.fivecell_coords == 0.0] = 0.000000001
        self.fivecell_pers = self.generate_persistence(self.fivecell_coords)

        self.sevencell = deepcopy(self.structure) 
        self.sevencell.make_supercell([7, 7, 7])
        self.sevencell_coords = self.sevencell.cart_coords
        self.sevencell_pers = self.generate_persistence(self.sevencell_coords)

        
        if write_out:
            pk.dump([self.threecell_pers, self.fivecell_pers, self.sevencell_pers], open(f"{output_path}_357_pd.pk", "wb"))

        exp_inf_pers = [None] * len(self.fivecell_pers)

        for i, _ in enumerate(self.fivecell_pers):
            exp_inf_pers[i] = PersistenceLimits(self.threecell_pers[i], self.fivecell_pers[i], self.sevencell_pers[i]).exp_inf

            print(f"Dim {i} infinite limit calculated")

        self.inf_pers = exp_inf_pers

        if write_out:
            self.write_to_file(self.inf_pers, self.output_path)

    def generate_lattice(self, cif, namespace):
        ''' Return lattice parameters from the cif file as a dict '''
        lattice = {}
        lattice['a'] = cif[namespace]['_cell_length_a']
        lattice['b'] = cif[namespace]['_cell_length_b']
        lattice['c'] = cif[namespace]['_cell_length_c']
        lattice['alpha'] = cif[namespace]['_cell_angle_alpha']
        lattice['beta'] = cif[namespace]['_cell_angle_beta']
        lattice['gamma'] = cif[namespace]['_cell_angle_gamma']
        return lattice

    def dio_pers(self, coords):
        simplices = fill_alpha_shapes(coords)
        f = Filtration(simplices)
        m = homology_persistence(f)
        dgms = init_diagrams(m, f)
        p_list = [[], [], []]
        for i, dgm in enumerate(dgms):
            if i > 2:
                break
            for pt in dgm:
                p_list[i].append((pt.birth, pt.death))
                # c_list[i]
                print(i, pt.birth, pt.death)

        pd = [np.array(dim) for dim in p_list]

        return pd

    def generate_persistence(self, xyz_coords):
        '''
        Apply the given filtration/complex (default alpha) and return the
        persistence diagrams of each dimension
        Currently using gudhi
        '''
        if self.complex == 'alpha':
            # Compute the alpha complex and persistence of the pointcloud
            alpha_complex = gd.alpha_complex.AlphaComplex(points=xyz_coords)
            simplex_tree = alpha_complex.create_simplex_tree()
            persistence = simplex_tree.persistence()

            # Convert gudhi style list into a k-dimensional list
            dimensions = max([x for (x, (y)) in persistence]) + 1
            pers = []
            for i in range(dimensions):
                dim_points = [np.array(x) for (y, (x)) in persistence if y == i]
                pers.append(np.vstack(dim_points))

            self.round_persistence(pers)
            self.remove_noise(pers)

            return pers

        elif self.complex == 'rips':
            # Compute the rips complex and persistence of the pointcloud
            rips_complex = gd.RipsComplex(points=xyz_coords)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
            persistence = simplex_tree.persistence()

            # Convert gudhi style list into a k-dimensional list
            dimensions = max([x for (x, (y)) in persistence]) + 1
            pers = []
            for i in range(dimensions):
                dim_points = [np.array(x) for (y, (x)) in persistence if y == i]
                pers.append(np.vstack(dim_points))

            # Round to reduce floating point errors, then remove any points on
            # the x=y line
            self.round_persistence(pers)
            self.remove_noise(pers)

            return pers

    def round_persistence(self, pers):
        """Use numpy.around to round to DECIMAL_ROUNDING places"""
        for i in range(len(pers)):
            pers[i] = np.around(pers[i], self.DECIMAL_ROUNDING)
        return pers

    def remove_noise(self, pers):
        '''
        Take each coordinate of a persistence list, and
        remove any points that have the same birth/death time

        TODO: Look into turning this into a similarity percentage and whether
        this affects the accuracy
        '''
        for i, dim in enumerate(pers):
            dim = np.array([x for x in dim if x[0] != x[1]])
            pers[i] = dim
        return pers

    def write_to_file(self, pers, output_path):
        '''Pickle and write all persistence points to a file'''
        try:
            pk.dump(pers, open(output_path, "wb"))

        except Exception as e:
            print(f"{self.output_path} failed to write to file in CifToPers.py because of {e}")

################################################################################
###############   Deprecated Code from here, TODO cleanup  #####################
################################################################################
    def new_persistence(self, expansion_factor):
        '''
        Take our cell, expand it by expansion_factor and recalculate the coords
        and the persistence (rounding to DECIMAL_ROUNDING decimal places)
        '''
        _, expanded_coords = self.generate_supercell(expansion_factor)

        pers = self.generate_persistence(expanded_coords)
        pers = self.round_persistence(pers)

        self.normalise_coords(pers)
        self.expanded_coords = expanded_coords

        return pers, expanded_coords

    def increment_expansion(self, xyz):
        '''
        Keep incrementing through the dimension in the expansion factor until
        incrementing no longer increases the number of persistence points
        '''
        while True:
            xyz = [x + 1 for x in xyz] # Increment our expansion factor
            new_pers, new_coords = self.new_persistence((xyz))

            if len(new_pers) > len(self.pers): # If there's a new homology dimension update
                self.pers, self.expanded_coords = new_pers, new_coords

            else:
                for j in range(len(self.pers)): # Else if theres a new persistence point in any dimension
                    if len(np.unique(self.pers[j])) < len(np.unique(new_pers[j])):
                        self.pers, self.expanded_coords = new_pers, new_coords

                    else:
                        self.pers, self.expanded_coords = new_pers, new_coords
                        return xyz

            if xyz[0] == self.MAX_EXPANSION: # else if we have reached max expansion factor
                break

        return xyz

    def find_minimal_expansion(self):
        '''
        Increment expansion in all 3 dimensions until we get the maximal
        expansion
        '''
        xyz = self.INITIAL_EXPANSION

        # Keep incrementing supercells until algorithm converges
        xyz = self.increment_expansion(xyz)
        self.xyz_expansion = xyz

        self.remove_noise(self.pers)

    

    

    def remove_zero_freq(self, counter_list):
        for dim in counter_list:
            for point in list(dim.keys()):
                if round(dim[point], self.DECIMAL_ROUNDING) < 0.000001:
                    dim.pop(point)

    def normalise_coords(self, pers):
        """
        Use PersistenceNormaliser to give the ratio dictionary of persistence
        points
        """
        normed_points = PersistenceNorm(pers)
        return normed_points.norm_list

    def flow_dist(self, comp2, comp1=None):
        """
        Take a second CifToPers object or the path to a cif file, process it
        and calculate the fractional bottleneck distance between the two
        """
        if comp1 == None:
            comp1 = self.inf_pers

        dist = PersistenceNorm().flow_norm_bottleneck(comp1, comp2)
        return dist

if __name__ == '__main__':
    main()

