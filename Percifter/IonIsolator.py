"""
Author: Cameron Hargreaves

A class to take in a cif file and split it into its' constituent anion, cations
and neutrally charged particles then write each of these to a separate cif
file
"""

import os
from CifFile import CifFile, ReadCif
import re
import pickle as pk

def main():
    testpath = '/home/cameron/Datasets/ICSD/ICSD_2019_Li_CIFs/icsd_015473.cif'
    output_path = '/home/cameron/Documents/tmp/icsd_015473/'
    x = ElementIsolator(testpath, output_path)

class IonIsolator():
    """
    Similar to Element Isolator except we simply split the compound into positively,
    and negatively charged ions
    """
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

        elements_in_cif = cif["_atom_site_label"]
        charged_elements = cif['_atom_type_symbol']

        cations, anions, neutrals = self.get_ions(charged_elements)

        # Remove all numeric values from elements_sum leaving chars to search on
        regex = '[A-Z][a-z]?'
        elements = list(re.finditer(regex, elements_sum, re.MULTILINE))

        # Empty array to store values
        isolated_cifs = [None] * 3

        for i, ions in enumerate([cations, anions, neutrals]):
            # Create a deepcopy without implementing the deepcopy methods, needed
            # due to pythons pass by object reference
            modified_cif = ReadCif(filepath)

            # Isolate the atom sites for each ion
            if '_atom_site_label' in cif:
                isolated_cifs[i] = (self.isolate_ions(modified_cif,
                                                    self.namespace,
                                                    '_atom_site_type_symbol',
                                                    ions))

            # Additionally remove aniso labels if these exist
            if '_atom_site_aniso_label' in cif:
                isolated_cifs[i] = (self.isolate_ions(modified_cif,
                                                    self.namespace,
                                                    '_atom_site_aniso_type_symbol',
                                                    ions))
            # isolated_cifs[i] = modified_cif

        self.isolated_cifs = isolated_cifs

        if output_path:
            self.write_to_file(isolated_cifs, output_path + self.icsd_code + "/")
        print()

    def isolate_ions(self, cif, namespace, loopLabel, searchTerms):
        if len(searchTerms) < 1:
            return None

        # take all keys containing atom_site
        atomSiteKeys = cif[namespace].GetLoop(loopLabel).keys()
        # get the positions of each of the atoms
        atomSiteLabel = cif[namespace][loopLabel]

        # Basic approach breaks down if you have two elements that share letters
        # eg: C and Cs so we strip the numeric chars from the cif entry and search
        # on the element name exactly
        elementIndex = []
        for i, atom_string in enumerate(atomSiteLabel):
            if atom_string in searchTerms:
                elementIndex.append(i)

        featureList = []
        # make a copy of keys, without copying, atomSiteKeys gets updated on
        # removal of each key which breaks the loop
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

    def get_ions(self, charged_elements):
            cations = []
            anions = []
            neutral = []

            for ion in charged_elements:
                if ion[-2] == "0" or ion[-1] == "0":
                    neutral.append(ion)
                elif ion[-1] == "-":
                    anions.append(ion)

                elif ion[-1] == "+":
                    cations.append(ion)

            return cations, anions, neutral

    def write_to_file(self, cifs, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i, ion in enumerate(["cations", "anions", "neutrals"]):
            print(ion)
            if cifs[i] != None:
                filename = self.icsd_code + "_" + ion + ".cif"
                outfile = open(output_path + filename, "w")
                outfile.write(cifs[i].WriteOut())
                outfile.close()

if __name__ == "__main__":
    main()
