"""
A class to take in a cif file and split it into its' constituent anion, cations
and neutrally charged particles OR into each constituent element and write each
of these to a separate output cif file. Can be used standalone to batch process
folders, or used in other programs directly

Useage:

# Write a cif to a new set of cif files and split by ion
CifIsolator(input_path, output_path, splitting_type="ion")

# Keep isolated cifs in memory and split by element
CifIsolator(input_path, splitting_type="element")
for cif_file in CifIsolator.isolated_cifs:
    do_something_with(cif_file)

Copyright (C) 2019  Cameron Hargreaves, Wenkai Zhang

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

import os
from CifFile import CifFile, ReadCif
import re
import pickle as pk

def main():
    testpath = '../tests/testfiles/icsd_000393.cif'
    output_path_ions = '../tests/testfiles/'
    output_path_elements = '../tests/testfiles/'

    x = CifIsolator(testpath, output_path_ions)
    x = CifIsolator(testpath, output_path_elements, splitting_type="element")
    print()

class CifIsolator():
    """
    A class to take in a cif file and split it into its' constituent anion, cations
    and neutrally charged particles OR into each constituent element and write each
    of these to a separate output cif file. Can be used standalone to batch process
    folders, or used in other programs directly

    Useage:

    # Write a cif to a new set of cif files and split by ion
    CifIsolator(input_path, output_path, splitting_type="ion")

    # Keep isolated cifs in memory and split by element
    CifIsolator(input_path, splitting_type="element")
    for cif_file in CifIsolator.isolated_cifs:
        do_something_with(cif_file)
    """

    def __init__(self, filepath, output_path=None, splitting_type="ion"):
        self.filepath = filepath
        self.output_path = output_path
        self.splitting_type = splitting_type

        self.icsd_code = self.filepath.split('/')[-1][:-4]

        self.cifFile = ReadCif(filepath)
        self.namespace = list(self.cifFile.keys())[0]
        self.cif = self.cifFile[self.namespace] # Simplify as we use this a lot


        self.formula = self.cif['_chemical_formula_structural']
        self.elements_sum = self.cif['_chemical_formula_sum']

        charged_elements = self.cif['_atom_type_symbol']
        cations, anions, neutrals = self.get_ions(charged_elements)

        # Remove all numeric values from elements_sum leaving chars to search on
        self.regex = '[A-Z][a-z]?'
        elements = list(re.finditer(self.regex, self.elements_sum, re.MULTILINE))
        self.elements = [element.group() for element in elements]

        if self.splitting_type == "ion":
            # Empty array to store values
            isolated_cifs = [None] * 3
            for i, ion in enumerate([cations, anions, neutrals]):
                self.isolate(i, ion, filepath, isolated_cifs, self.isolate_ions)

        if self.splitting_type == "element":
            # Empty array to store values
            isolated_cifs = [None] * len(self.elements)
            for i, element in enumerate(self.elements):
                self.isolate(i, element, filepath, isolated_cifs, self.isolate_elements)

        self.isolated_cifs = isolated_cifs

        if output_path:
            self.write_to_file(isolated_cifs, output_path + self.icsd_code + "/")

        print()

    def isolate(self, i, ion, filepath, isolated_cifs, isolation_function):
        """
        Create modified cif files for each of the isolation types
        """
        # Create a new copy of the file
        modified_cif = ReadCif(filepath)

        # Isolate the atom sites for each ion
        if '_atom_site_label' in self.cif:
            isolated_cifs[i] = (isolation_function(modified_cif,
                                                self.namespace,
                                                '_atom_site_type_symbol',
                                                ion))

        # Additionally remove aniso labels if these exist
        if '_atom_site_aniso_label' in self.cif:
            isolated_cifs[i] = (isolation_function(modified_cif,
                                                self.namespace,
                                                '_atom_site_aniso_type_symbol',
                                                ion))

    def isolate_ions(self, cif, namespace, loopLabel, searchTerms):
        """
        For each set of anions/cations return an isolated cif file
        """
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
        """
        Split a list of elements and their charges and split into separate lists
        """
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

    def isolate_elements(self, cif, namespace, loopLabel, searchTerm):
        """
        For a given cif file return an isolated cif file for an element
        """
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

    def write_to_file(self, cifs, output_path):
        """
        Writeout a cif file labelled by the ion/element isolated
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.splitting_type == "ion":
            for i, ion in enumerate(["cations", "anions", "neutrals"]):
                if cifs[i] != None:
                    filename = self.icsd_code + "_" + ion + ".cif"
                    outfile = open(output_path + filename, "w")
                    outfile.write(cifs[i].WriteOut())
                    outfile.close()

        elif self.splitting_type == "element":
            for i, element in enumerate(self.elements):
                if cifs[i] != -1:
                    filename = self.icsd_code + "_" + element + ".cif"
                    outfile = open(output_path + filename, "w")
                    outfile.write(cifs[i].WriteOut())
                    outfile.close()

if __name__ == "__main__":
    main()
