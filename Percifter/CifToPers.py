'''
This class takes an input cif file, performs an expansion in x, y, z dimensions
and then returns a persistence diagram if we have minimal change on expansions
by writing to the given filepath

In this file we are experimenting with using ripser to see how this compares to
gudhi

Parameters
cif_path : Path to search for cif file
output_path : Path to write final persistence diagram to
'''
import os
import time
from collections import Counter

import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

import CifFile

# Difficulties in setting up these two libraries, they may be re-added in future version
# from ripser import ripser
# import cechmate as cm
import gudhi as gd

from diffpy.structure import loadStructure
from diffpy.structure.expansion.supercell_mod import supercell
from sklearn.metrics.pairwise import euclidean_distances

from Niggli import Niggli
from PersistanceNormaliser import PersistenceNorm

# Ripser/matplotlib throw a lot of annoying warnings!
import warnings
#warnings.filterwarnings("ignore")

class CifToPers():
    def __init__(self, input_path=None,
                       output_path=None,
                       simplicial_complex='alpha',
                       ROUND=True,
                       DECIMAL_ROUNDING=5,
                       REMOVE_NOISE=True,
                       SIMILARITY_PERCENTAGE=0.05,
                       INITIAL_EXPANSION=[2, 2, 2],
                       MAX_EXPANSION=10):

        self.DECIMAL_ROUNDING = DECIMAL_ROUNDING
        self.SIMILARITY_PERCENTAGE = SIMILARITY_PERCENTAGE
        self.MAX_EXPANSION = MAX_EXPANSION
        self.INITIAL_EXPANSION = INITIAL_EXPANSION

        self.cif_path = input_path
        self.complex = simplicial_complex


        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = input_path[:-4] + ".pers"

        self.cif = CifFile.ReadCif(input_path)

        namespace = list(self.cif.keys())[0]
        self.keys = list(self.cif[namespace].keys())

        self.lattice = self.generate_lattice(self.cif, namespace)
        # TODO implement the Niggli reduction
        # self.niggli_lattice = self.reduce_niggli(self.lattice)

        self.diffpy_molecule = loadStructure(input_path)
        self.xyz_coords = self.diffpy_molecule.xyz_cartn
        self.expanded_cell, self.expanded_coords = self.generate_supercell(INITIAL_EXPANSION)

        self.pers = self.generate_persistence(self.expanded_coords)

        # Iterate until we find the expansion where we no longer get increased
        # Persistence points, and update self.pers
        self.find_minimal_expansion()

        # We generate a lot of noise from floating point errors so remove these
        self.remove_noise(self.pers)

        # Use PersistenceNorm to calculate fractional persistence points
        self.normalise_coords(self.pers)

        self.write_to_file(self.pers, self.output_path)

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

    def reduce_niggli(self, lattice, epsilon):
        '''
        Take the original six lattice parameters and follow Grubers algorithm
        to generate the Niggli cell
        '''
        return Niggli(lattice).niggli_lattice

    def calc_niggli_expansion(self, lattice, niggli_lattice):
        '''
        Generate a supercell that encompasses the Niggli cell. Will probably use
        Niggli class
        TODO FINISH THIS FUNCTION
        '''

    def generate_supercell(self, expansion):
        '''
        Use diffpy to create a new supercell via dimensions [x, y, z]
        Parameter: expansion, list of integers
        '''
        expandedCell = supercell(self.diffpy_molecule, expansion) # Expand
        coords = np.array(expandedCell.xyz_cartn) # Take this as a numpy array
        return expandedCell, coords

    def generate_persistence(self, xyz_coords):
        '''
        Apply the given filtration/complex (default alpha) and return the
        persistence diagrams of each. Default to first homology group only.
        Currently using gudhi
        TODO: Cleanup, maybe all use gudhi?
        '''
        if self.complex == 'alpha':
            # Compute the alpha complex and persistence of the pointcloud
            alpha_complex = gd.AlphaComplex(points=xyz_coords)
            simplex_tree = alpha_complex.create_simplex_tree()
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

    def write_to_file(self, pers, output_path):
        '''
        Pickle and write all persistence points to a file
        '''
        try:
            pk.dump(pers, open(output_path, "wb" ) )

        except Exception as e:
            print(f"{self.filename} failed to write to file in CifToPers.py because of {e}")

    def round_persistence(self, pers):
        """Use numpy.around to round to DECIMAL_ROUNDING places"""
        for i in range(len(pers)):
            pers[i] = np.around(pers[i], self.DECIMAL_ROUNDING)
        return pers

    def remove_noise(self, pers):
        '''
        Take each coordinate of a persistence list, round to decimal_places and
        remove any points that have the same birth/death time

        TODO: Look into turning this into a similarity percentage and whether
        this affects the accuracy
        '''
        for i, dim in enumerate(pers):
            dim = np.array([x for x in dim if x[0] != x[1]])
            pers[i] = dim
        return pers


        # CURRENTLY UNUSED
        # Remove any points that are 0.5% similar as these have a short lifespan
        # similar_points = []
        # lower_bound = 1 - self.SIMILARITY_PERCENTAGE / 2
        # upper_bound = 1 + self.SIMILARITY_PERCENTAGE / 2
        # for point in frequency_map: # loop through keys
        #     x, y = point[1][0], point[1][1] # Take birth and death
        #     if y >= x * lower_bound and y <= x * upper_bound: # If within 2% similarity remove
        #         similar_points.append(point)
        # for point in similar_points:
        #     frequency_map.pop(point)
        # return frequency_map # A dictionary of counts

    def normalise_coords(self, pers):
        """
        Use PersistenceNormaliser to give the ratio dictionary of persistence
        points
        """
        self.PersistenceNorm = PersistenceNorm(pers)
        self.normed_persistence = self.PersistenceNorm.norm_list

    def flow_dist(self, comp2, comp1=None):
        """
        Take a second CifToPers object or the path to a cif file, process it
        and calculate the fractional bottleneck distance between the two
        """
        if comp1 == None:
            comp1 = self.PersistenceNorm

        if type(comp2) == CifToPers:
            comp2 = comp2.PersistenceNorm

        elif os.path.exists(comp2):
            comp2 = CifToPers(comp2).PersistenceNorm

        return comp1.flow_norm_bottleneck(comp2)


if __name__ == '__main__':
    input_path = '/home/cameron/Dropbox/University/PhD/Percifter/Percifter/cifs/icsd_000003_anions.cif'
    test_path = '/home/cameron/Dropbox/University/PhD/Percifter/Percifter/cifs/icsd_000003_cations.cif'
    out_path = '/home/cameron/Documents/tmp/norm_dist/'

    x = CifToPers(input_path, out_path + "icsd_000003.pers")
    y = CifToPers(test_path, out_path + "icsd_000003_cations.pers")

    print(x.flow_dist(y))
