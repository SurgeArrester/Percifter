"""
This class takes in a cif file and returns equivalent cif files each contatining
a single element from the structural formula only, to be parsed by CifToPers

Author: Cameron Hargreaves
"""

import os
from CifFile import CifFile, ReadCif
import re
import pickle as pk

def main():
    testpath = '/home/cameron/Datasets/ICSD/ICSD_2019_Li_CIFs/icsd_015473.cif'
    output_path = '/home/cameron/Documents/tmp/icsd_015473/'
    x = ElementIsolator(testpath, output_path)

class ElementIsolator():
    def __init__(self, filepath, output_path=None):
        self.filepath = filepath
        self.output_path = output_path
        self.icsd_code = self.filepath.split('/')[-1][:-4]

        self.cifFile = ReadCif(filepath)
        self.namespace = list(self.cifFile.keys())[0]
        cif = self.cifFile[self.namespace] # Simplify as we use this a lot
        keys = list(cif.keys())

        self.formula = cif['_chemical_formula_structural']
        elements_sum = cif['_chemical_formula_sum']

        # Remove all numeric values from elements_sum leaving chars to search on
        regex = '[A-Z][a-z]?'
        elements = list(re.finditer(regex, elements_sum, re.MULTILINE))

        # Create empty array for cifs
        isolated_cifs = [None] * len(elements)

        for i, element in enumerate(elements):
            # Create a deepcopy without implementing the deepcopy methods
            modified_cif = ReadCif(filepath)

            # Isolate the atom sites for each element
            if '_atom_site_label' in cif:
                isolated_cifs[i] = (self.deleteAndUpdateLoop(modified_cif,
                                                    self.namespace,
                                                    '_atom_site_label',
                                                    element.group()))

            # Additionally remove aniso labels if these exist
            if '_atom_site_aniso_label' in cif:
                isolated_cifs[i] = (self.deleteAndUpdateLoop(modified_cif,
                                                    self.namespace,
                                                    '_atom_site_aniso_label',
                                                    element.group()))
            isolated_cifs[i] = modified_cif

        self.isolated_cifs = isolated_cifs
        self.elements = [element.group() for element in elements]

        if output_path:
            self.write_to_file(isolated_cifs, self.elements, output_path + self.icsd_code + "/")


    def deleteAndUpdateLoop(self, cif, namespace, loopLabel, searchTerm):
        # [key for key in keys if key.find('atom_site') > 0] # take all keys containing atom_site
        atomSiteKeys = cif[namespace].GetLoop(loopLabel).keys()
        # get the positions of each of the atoms
        atomSiteLabel = cif[namespace][loopLabel]

        # Basic approach breaks down if you have two elements that share letters
        # eg: C and Cs so we strip the numeric chars from the cif entry and search
        # on the element name exactly
        elementIndex = []
        for i, atom_string in enumerate(atomSiteLabel):
            atom_no_numeric = "".join(filter(lambda y: not y.isdigit(), atom_string))
            if atom_no_numeric == searchTerm:
                elementIndex.append(i)

        featureList = []
        # make a copy, without copy atomSiteKeys gets updated on removal of each key which breaks things
        keys = atomSiteKeys[:]
        for key in keys:
            featureList.append([cif[namespace][key][i] for i in elementIndex])
            cif[namespace].RemoveItem(key)

        # Add a loop block
        # if we have lithiums (crashes if none in anisotropic block)
        if elementIndex:
            cif[namespace].AddLoopItem((keys,
                                        featureList))

        return cif

    def write_to_file(self, cifs, elements, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i, element in enumerate(elements):
            print(element)
            if cifs[i] != -1:
                filename = self.icsd_code + "_" + element + ".cif"
                outfile = open(output_path + filename, "w")
                outfile.write(cifs[i].WriteOut())
                # x = cifs[i].WriteOut()
                outfile.close()

if __name__ == "__main__":
    main()
