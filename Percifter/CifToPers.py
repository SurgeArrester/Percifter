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

from .Niggli import Niggli

# Ripser/matplotlib throw a lot of annoying warnings!
import warnings
#warnings.filterwarnings("ignore")

class CifToPers():
    def __init__(self, input_path = None,
                       output_path = None,
                       filename = None,
                       simplicial_complex = 'alpha',
                       reduced_persistence = False,
                       DECIMAL_ROUNDING = 3,
                       SIMILARITY_PERCENTAGE = 0.05,
                       NORMALISE = False,
                       INITIAL_EXPANSION = [2, 2, 2],
                       MAX_EXPANSION = 10):

        self.reduced_persistence = reduced_persistence
        self.DECIMAL_ROUNDING = DECIMAL_ROUNDING
        self.SIMILARITY_PERCENTAGE = SIMILARITY_PERCENTAGE
        self.NORMALISE = NORMALISE
        self.MAX_EXPANSION = MAX_EXPANSION
        self.INITIAL_EXPANSION = INITIAL_EXPANSION

        self.cif_path = input_path
        self.output_path = output_path

        if filename:
            self.filename = filename
        else:
            print(input_path)
            # Take the name of the cif file from the path
            self.filename = input_path.split('/')[-1].split('.')[-2]

        self.complex = simplicial_complex

        self.cif = CifFile.ReadCif(input_path)

        namespace = list(self.cif.keys())[0]
        self.keys = list(self.cif[namespace].keys())

        # self.lattice = self.generate_lattice(self.cif, namespace)
        # self.niggli_lattice = self.reduce_niggli(self.lattice)

        try:
            self.diffpy_molecule = loadStructure(input_path)
            self.xyz_coords = self.diffpy_molecule.xyz_cartn
            self.expanded_cell, self.expanded_coords = self.generate_supercell((3, 3, 3))

            if self.NORMALISE:
                self.expanded_coords = self.normalise_coords(self.expanded_coords)

            self.pers = self.generate_persistence(self.expanded_coords)

            # Iterate until we find the expansion where we no longer get increased
            # Persistence points
            self.find_minimal_expansion()
            self.write_to_file(self.pers, output_path, reduced_persistence)

        except Exception as e:
            with open("/home/cameron/Datasets/ICSD/MineralClass/failed_cif", "a+") as myfile:
                myfile.write(f"{self.filename} failed because of {e}\n")
                myfile.close()
            print(f"Failed cif file, check the formatting against others or get in touch with C.J.Hargreaves@Liverpool.ac.uk\n{e}")

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
        persistence diagrams of each. Default to first homology group only. Currently
        using three different libraries
        TODO: Cleanup, maybe all use gudhi?
        '''
        if self.complex == 'alpha':
            alpha_complex = gd.AlphaComplex(points=xyz_coords)
            simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=60.0) # Essentially a delaunay triangulation
            persistence = simplex_tree.persistence()

            # Convert this long list into a k-dimensional array for each dimension
            dimensions = max([x for (x, (y)) in persistence])
            pers = []
            for i in range(dimensions):
                dim_points = [np.array(x) for (y, (x)) in persistence if y == i]
                pers.append(np.vstack(dim_points))
            return pers

        # TODO re-test the other complexes
        # elif self.complex == 'cech':
        #     cech = cm.Cech(max_dim=1)
        #     cech.build(xyz_coords)
        #     return cech.diagrams()

        # elif self.complex == 'rips':
        #     pers = ripser(xyz_coords, maxdim=1)
        #     return pers['dgms'] # return diagram

    def new_persistence(self, expansion_factor):
        '''
        Take our cell, expand it by expansion_factor and recalculate the coords
        and the persistence (removing values close to x=y and rounding to 8 dp)
        '''
        _, expanded_coords = self.generate_supercell(expansion_factor)
        if self.NORMALISE:
            expanded_coords = self.normalise_coords(expanded_coords)
        pers = self.generate_persistence(expanded_coords)
        # pers = self.reduce_persistence(pers, self.DECIMAL_ROUNDING)
        return pers, expanded_coords

    def increment_expanion(self, i, xyz):
        '''
        Keep incrementing through the dimension in the expansion factor until
        incrementing no longer increases the number of persistence points
        '''
        while True:
            xyz[i] += 1 # Increment our expansion factor
            new_pers, new_coords = self.new_persistence((xyz))
            if len(new_pers) > len(self.pers): # If there's a new homology dimension update
                self.pers, self.expanded_coords = new_pers, new_coords

            for j in range(len(self.pers)): # Else if theres a new persistence point in any dimension
                if len(np.unique(self.pers[j])) < len(np.unique(new_pers[j])):
                    self.pers, self.expanded_coords = new_pers, new_coords

            if len(np.unique(self.pers[0])) == len(np.unique(new_pers[0])):
                # Adding in an extra dimension hasn't gotten more points
                break

            elif xyz[i] < self.MAX_EXPANSION: # else if we have reached max expansion factor
                break
        return xyz

    def find_minimal_expansion(self):
        '''
        Increment expansion in all 3 dimensions until we get the maximal
        expansion
        '''
        xyz = self.INITIAL_EXPANSION       # set initial expansion to 1x1x1
        for i in range(3):    # Loop through each dimension
            xyz = self.increment_expanion(i, xyz)
        self.xyz_expansion = xyz
        print(self.filename)
        print("XYZ Expansion factor is: {}".format(xyz))
        for i in range(len(self.pers)):
            print(f"Homology dimension {i} has unique points: {np.unique(self.pers[i])}") # just the unique points

    def write_to_file(self, pers, output_path, reduction):
        '''
        Use a Counter to reduce the number of persistence points to their unique
        values
        '''
        try:
            if reduction:
                for i in range(len(pers)):
                    x = map(tuple, pers[i]) # np array is unhashable so must first be cast to a list
                    pers[i] = Counter(x) # Use each coordinate as a key in a dictionary and then count occurences
                pk.dump(pers, open(output_path, "wb" ) )

            else:
                pk.dump(pers, open(output_path, "wb" ) )

        except Exception as e:
            print(f"{self.filename} failed to write to file in CifToPers.py because of {e}")

    def reduce_persistence(self, pers, decimal_places):
        '''
        UNUSED, may be implemented to round the persistence points to certain
        number of decimal places

        Take each coordinate of a persistence list, round to decimal_places and
        remove any points that are within 2% similarity
        '''
        rounded_list = [(dim, tuple(np.around(coords, decimals=decimal_places)))
                            for (dim, coords) in pers]
        frequency_map = Counter(rounded_list)

        # Remove any points that are 0.5% similar as these have a short lifespan
        similar_points = []
        lower_bound = 1 - self.SIMILARITY_PERCENTAGE / 2
        upper_bound = 1 + self.SIMILARITY_PERCENTAGE / 2
        for point in frequency_map: # loop through keys
            x, y = point[1][0], point[1][1] # Take birth and death
            if y >= x * lower_bound and y <= x * upper_bound: # If within 2% similarity remove
                similar_points.append(point)
        for point in similar_points:
            frequency_map.pop(point)
        return frequency_map # A dictionary of counts

    
    def normalise_coords(self, coords):
        """
        UNUSED PRESENTLY
        Normalise distances by making the smallest distance of the coordinates
        equal to 1
        """
        dist = euclidean_distances(coords, coords) # make distance matrix
        mindist = np.min(dist[np.nonzero(dist)]) # find the smallest nonzero distance
        normed_coord = coords / mindist # scale by this to make the shortest distance 1
        return normed_coord

if __name__ == '__main__':
    input_path = '/home/cameron/Documents/tmp/icsd_977903/icsd_977903_Fe.cif'
    test_folder = '/home/cameron/Documents/tmp/icsd_977903/'
    out_path = '/home/cameron/Documents/tmp/icsd_977903/icsd_977903_Fe.pers'

    x = CifToPers(input_path, out_path)
