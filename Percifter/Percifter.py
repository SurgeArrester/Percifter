import os
import warnings 
import signal 
import time
import pickle as pk

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt 

from ripser import Rips

from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix

import ase
from ase.io import read as read_cif
from ase.build import cut as make_supercell
from ase import Atoms

from ElMD import EMD

def main():
    cif_folder = r"./TestFiles/Ruddlesden/"
    pers_folder = r"./TestFilesPers/"

    os.chdir("/home/cameron/Dropbox/University/FourthYear/Percifter")

    my_filenames = np.array(sorted(os.listdir(cif_folder)), dtype=object)

    for f in my_filenames:
        print(f)
        f_path = os.path.join(pers_folder, f[:-4] + ".pers")
        if os.path.exists(f_path):
            x = Percifter()
            x.load(f_path)
        else:
            x = Percifter(cif_folder + f, verbose=True, reduce_cell=False)

        print(x.pers)
        x.save(f_path)

class Percifter():
    def __init__(self, 
                path=None, 
                n_dim=1, 
                isolate=None, 
                expansions=[2, 3, 4],
                interval_dp = 3,
                x_y_tolerence=0.03, 
                ripser_error_tolerence=2, 
                converged_tolerence=0.0001,
                timeout=600, 
                verbose=True, 
                reduce_cell=False):
        self.path = path # The cif path to read
        self.n_dim = n_dim # The largest homology group to go up to
        self.isolate = isolate # An element, or list of elements, to take the isolated position
        self.expansions = expansions # The unit cell expansions to take
        self.interval_dp = interval_dp # The dp to round each persistence interval
        self.x_y_tolerence = x_y_tolerence # points close to the x=y axis are often noise
        self.ripser_error_tolerence = ripser_error_tolerence # Ripser can introduce some incorrect intervals
        self.converged_tolerence = converged_tolerence # The limit above which we include features in the CPD
        self.timeout = timeout # These can take a long time to process
        self.reduce_cell = reduce_cell # Whether to first perform a lattice reduction (this often introduced errors when testing)

        self.primitive_cell = None
        self.structure = [None, None, None]
        self.points = [None, None, None]
        self.verbose = verbose

        self.processing_time = None
        
        if path is not None:
            time_start = time.time()
            self.read_points(path)
            self.pers = self.ConvergentPers(n_dim, self)
            self.process()
            self.processing_time = time.time() - time_start

    def __repr__(self):
        # ASE overwrites anion positions, so this can miss elements from the
        # compositions...
        return f"Percifter(formula={self.primitive_cell.symbols})"

    def load(self, fname):
        with open(fname, 'rb') as f:
            tmp_dict = pk.load(f)

        self.__dict__.update(tmp_dict)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pk.dump(self.__dict__, f)

    def read_points(self, fhandle, inp_type="cif"):
        ase_lattices =  tuple(cls for cls in ase.lattice.__dict__.values() if isinstance(cls, type))
        if isinstance(fhandle, Atoms):
            self.primitive_cell = fhandle

        elif inp_type == "cif":
            self.primitive_cell = read_cif(fhandle, subtrans_included=False, primitive_cell=self.reduce_cell)

        if self.verbose: print("Making Supercells")

        self.primitive_cell.pbc[2] = False
        for i, j in enumerate(self.expansions):
            self.structure[i] = make_supercell(self.primitive_cell, a=(j, 0, 0), b=(0, j, 0), c=(0, 0, j))

            # Take only the points of the specified elements
            if self.isolate is not None:
                if isinstance(self.isolate, str):
                    bool_array = self.structure[i].symbols == self.isolate
                elif isinstance(self.isolate, list):
                    bool_array = np.where(np.isin(self.structure[i].symbols, self.isolate))

            # If no isolated elements take all points
            else:
                bool_array = range(len(self.structure[i].get_positions()))

            self.points[i] = self.structure[i].get_positions()[bool_array]

    def process(self):
        self._generate_pers()
        self._clean_pers()
        self._normalise_pers()
        self._find_limits()
        self._remove_error()
        # self.pers.link_cycles() # tends not to work in practice :(

    def emd(self, y, x=None, average=True):
        """
        Compute the EMD between the CPDs of this object and a second Percifer object

        If average=True then will return the average across all homology groups
        but otherwise will return a list of distances for each homology group
        """
        if not isinstance(y, Percifter):
            raise TypeError("Input must be a Percifter object")

        if x is None:
            x = self

        distances = []

        for i, diag in enumerate(x.pers.inf_pers):
            if len(diag) == 0 or len(y.pers.inf_pers[i]) == 0:
                distances.append(None)
            
            else:
                x_c, x_f = zip(*[(component, fraction) for component, fraction in diag.items()])
                y_c, y_f = zip(*[(component, fraction) for component, fraction in y.pers.inf_pers[i].items()])

                dm = distance_matrix(x_c, y_c)

                distances.append(EMD(np.array(x_f), np.array(y_f), dm))

        if average:
            return np.mean([x for x in distances if x is not None])
        else:
            return distances


    def plot(self, display=True):
        labels = ["Connected Components", "Cycles"]
        plt.axvline(x=0, color="black", linestyle="--")
        
        for i, diagram in enumerate(self.pers):
            points = np.array(list(diagram.keys()))
            # print(points)
            plt.scatter(points[:, 0], points[:, 1], label=labels[i])

        plt.plot([0, plt.ylim()[1]], [0, plt.ylim()[1]], '--', color='k')
        plt.legend(loc='lower right')
        
        plt.xlim(-0.2, plt.xlim()[1])
        plt.ylim(bottom=0)
        
        plt.axis("equal")

        if display: plt.show()

    def _timeout_handler(self, num, stack):
        raise TimeoutError("Timeout")

    def _generate_pers(self):
        """
        From the generated pointsets for each cell expansion, generate the 
        associated persistence diagram. If this exceeds the timeout period then 
        raise an exception
        """
        for i, pointset in enumerate(self.points):
            if self.verbose: print(f"Generating PH dim={self.expansions[i]}, points={len(pointset)}")

            # Compute the rips complex and filtration of the pointcloud
            rips_complex = Rips(maxdim=self.n_dim, verbose=self.verbose)
            
            # Set a timeout as this can take a very long time for many points
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout)

            try:
                intervals = rips_complex.fit_transform(pointset)
                self.pers.diag[i] = intervals
                # self.pers.cycles[i] = cycles

            except TimeoutError:
                raise TimeoutError("Exceeded available time limit, try increasing self.timeout")

            finally:
                # Cancel the previously set alarm
                signal.alarm(0)

    def _clean_pers(self):
        """Remove the artifacts from computational geometry"""
        # For each expansion
        for expansion, diagrams in enumerate(deepcopy(self.pers.diag)):
            # For each diagram
            for i, diag in enumerate(diagrams):
                # Round these points to 2dp
                # print(diag)
                self.pers.diag[expansion][i] = np.round(self.pers.diag[expansion][i], self.interval_dp)
                
                # After rounding we shouldn't have any unique points, and any which
                # are there are usually from computational arithmetic error
                vals, inds, counts = np.unique(self.pers.diag[expansion][i], axis=0, return_index=True, return_counts=True)
                
                # At present, ripser.py returns some erroneous features seemingly
                # at random. If there are fewer than this number remove them
                # TODO is this caught by self.tolerence below anyway?
                inds = inds[np.where(counts < self.ripser_error_tolerence)[0]]
                self.pers.diag[expansion][i] = np.delete(self.pers.diag[expansion][i], inds, axis=0)
                
                # if len(self.pers.cycles[expansion][i]) > 0:
                #     self.pers.cycles[expansion][i] = np.delete(self.pers.cycles[expansion][i], inds, axis=0)

                # Remove any points which are very close to the x=y axis 
                self.pers.diag[expansion][i] = np.array([p for p in self.pers.diag[expansion][i] if not np.allclose(p[0], p[1])])

                # Finally remove any points which are out of our error x_y_tolerence
                self.pers.diag[expansion][i] = np.array([p for p in self.pers.diag[expansion][i] if (p[1] - p[0]) / p[1] > self.x_y_tolerence])

    def _normalise_pers(self):
        '''For each of the diagrams, normalize these to get the ratio'''
        # Copy over diagrams, will be overwritten
        self.pers.normed_diag = deepcopy(self.pers.diag) 

        # For each expansion
        for i, expansion in enumerate(self.pers.diag):
            # For each diagram
            # Strip the infinite points
            for j, dim in enumerate(expansion):
                if not dim.any():
                    self.pers.normed_diag[i][j] = dim 

                else:
                    self.pers.diag[i][j] = dim[~np.isinf(dim).any(axis=1)]

            # Cast to hashable tuples so we may apply the Counter
            for j, dim in enumerate(expansion):
                summed_diag = Counter([tuple(x) for x in dim])
                total = sum(summed_diag.values(), 0.0)

                # Normalize the point ratios
                for key in summed_diag:
                    summed_diag[key] /= total

                self.pers.normed_diag[i][j] = summed_diag

    def _find_limits(self, n="inf"):
        ''' Find the limits of our persistence diagrams at expansion n'''
        xs = np.array(self.expansions)

        # Repeat for each homology group
        for dimension in range(self.n_dim + 1):
            exp_n = Counter()

            for i, expansion in enumerate(xs):
                # loop through each point in our diagram
                for point, frequency in self.pers.normed_diag[0][dimension].items():
                    # Collate the points frequency at each expansion
                    ys = [frequency, 
                          self.pers.normed_diag[1][dimension][point], 
                          self.pers.normed_diag[2][dimension][point]]

                    # Derive the convergence function
                    func = self._get_base_func(xs, ys)

                    # Calculate the optimal final value (scipy will always throw
                    # a warning as we cannot calculate covariance of our values 
                    # with only three points)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        popt, pcov = curve_fit(func, xs, ys, bounds=(0, 1))

                    # Return converged value
                    if n == "inf" and func == self._const_func:
                        exp_n[point] = popt[0]
                    elif n == "inf":
                        exp_n[point] = popt[2]
                    elif type(popt) == tuple:
                        exp_n[point] = func(n, *popt[0])
                    else:
                        exp_n[point] = func(n, *popt)

                self.pers.inf_pers[dimension] = exp_n

    def _get_base_func(self, xs, ys):
        '''
        Check if a series is stable, descending to a limit, or ascending to a 
        limit and return the appropriate function.
        '''
        if ys[0] < ys[1]:
            return self._inv_recip_func

        elif ys[0] > ys[1]:
            return self._recip_func

        elif ys[0] == ys[1]:
            return self._const_func

    def _inv_recip_func(self, x, a, b, c):
        return (-a / (x - b)) + c

    def _recip_func(self, x, a, b, c):
        return (a / (x - b)) + c

    def _const_func(self, x, c):
        return c

    def _remove_error(self):
        '''
        The sum of all the ratios should sum to 1. There are unavoidable 
        errors in precision which is compounded when taking limits
        up to infinity. Here we remove this error and also remove all points 
        with zero frequency, then scale to give a mass of 1 if needed
        '''
        for dim in range(self.n_dim + 1):
            # Remove all points below the accepted ratio of points to include
            # in the CPD
            for k, v in list(self.pers.inf_pers[dim].items()):
                if v < self.converged_tolerence:
                    self.pers.inf_pers[dim].pop(k)

            # Ensure the new mass of points sums to 1
            total_rat = sum([v for k, v in self.pers.inf_pers[dim].items()])

            # The persistence diagram is empty so skip
            if total_rat == 0:
                continue
            
            # If the mass of the CPD is not 1
            while (not np.isclose(1 - total_rat, 0)):
                # Scale by the difference of ratio which should give a new sum of 1
                scaler = 1 / total_rat 

                for k, v in list(self.pers.inf_pers[dim].items()):
                    self.pers.inf_pers[dim][k] = v * scaler

                total_rat = sum([v for k, v in self.pers.inf_pers[dim].items()])

    class ConvergentPers():
        '''
        Subclass to represent Convergent persistence diagrams
        '''
        def __init__(self, n, model):
            self.n = n
            self.diag = [None for _ in range(3)]
            self.normed_diag = [None for _ in range(3)]
            self.cycles = [None for _ in range(3)]
            self.cycle_lookup = defaultdict(dict)
            self.inf_pers = [None for _ in range(n + 1)]
            self.ase_structures = model.structure

        def link_cycles(self):
            """
            Take each of the intervals and use these as a key in a lookup table
            to link the associated cycles with each interval. 

            This function works only with the ripser branch which returns the
            generators of each cycle, however the generators are not what you
            may think of as the "correct" path nor the shortest path, and are
            very variable in length making them difficult to link in practice
            without some more graph theory. 
            """
            expansion = 0

            for h_dim in range(self.n + 1):
                for i, cycle in enumerate(self.cycles[expansion][h_dim]):
                    if np.any(np.isnan(cycle)):
                        continue 

                    if len(self.diag[expansion][h_dim]) == 0 or len(self.diag[expansion][h_dim]) == i:
                        continue

                    if tuple(self.diag[expansion][h_dim][i]) not in self.inf_pers[h_dim]:
                        continue

                    if h_dim == 0:
                        elemental_cycle = [self.ase_structures[expansion].get_chemical_symbols()[int(node)] for node in cycle]

                        if elemental_cycle[0] == elemental_cycle[1]:
                            break

                    else:
                        elemental_cycle = [self.ase_structures[expansion].get_chemical_symbols()[int(node[0])] for node in cycle]

                    key = str(self.diag[expansion][h_dim][i])

                    if key in self.cycle_lookup[h_dim]:
                        self.cycle_lookup[h_dim][key].append(elemental_cycle)
                    else:
                        self.cycle_lookup[h_dim][key] = [elemental_cycle]


                for k, v in list(self.cycle_lookup[h_dim].items()):
                    self.cycle_lookup[h_dim][k] = Counter([tuple(x) for x in self.cycle_lookup[h_dim][key]])

        def __repr__(self):
            ret = "ConvergentPersDiag(Birth, Death): Freq\n"
            for i, diag in enumerate(self.inf_pers):
                ret += f"H{i}\t"

                # Add each interval and frequency to a return string
                for s in [str(interval) + ": " + f"{frequency:.4f}" + ", " for interval, frequency in diag.items()]:
                    ret += s

                # Strip the trailing comma and add a newline
                if len(list(diag.keys())) > 0:
                    ret = ret[:-2]

                ret += "\n"

            # Strip trailing newline
            return ret[:-1]

        def __getitem__(self, i):
            return self.inf_pers[i]

if __name__ == "__main__":
    main()