"""
This class takes in a cif file and returns equivalent cif files each contatining
a single element from the structural formula only, to be parsed by CifToPers

This was attempted using PyCifRw but there are bugs in the WriteOut() function
that I can't be bothered to track down, but have raised an issue on the repository
Hence the unused isolate_element() method
https://bitbucket.org/jamesrhester/pycifrw/issues/18/writeout-str-representation-appears-to-be

"""

import os
from CifFile import CifFile, ReadCif
import re
import pickle as pk

class ElementIsolator():
    def __init__(self, filepath, output_path=None, isolate="anions"):
        self.filepath = filepath
        self.output_path = output_path
        self.isolate = isolate
        self.icsd_code = self.filepath.split('/')[-1][:-4]

        self.cifFile = ReadCif(filepath)
        self.namespace = list(self.cifFile.keys())[0]
        cif = self.cifFile[self.namespace] # Simplify as we use this a lot
        keys = list(cif.keys())

        self.formula = cif['_chemical_formula_structural']
        elements_sum = cif['_chemical_formula_sum']

        elements_in_cif = cif["_atom_site_label"]
        elements_charge = cif['_atom_site_type_symbol']
        # Remove all numeric values from elements_sum leaving chars to search on
        regex = '[A-Z][a-z]?'
        elements = list(re.finditer(regex, elements_sum, re.MULTILINE))

        isolated_cifs = [-1] * len(elements)

        if self.isolate == "elements":
            for i, element in enumerate(elements):
                # Create a deepcopy without implementing the deepcopy methods
                modified_cif = ReadCif(filepath)

                # Isolate the atom sites for each element
                if '_atom_site_label' in cif:
                    isolated_cifs[i] = (self.isolate_element(modified_cif,
                                                        self.namespace,
                                                        '_atom_site_label',
                                                        element.group()))

                # Additionally remove aniso labels if these exist
                if '_atom_site_aniso_label' in cif:
                    isolated_cifs[i] = (self.isolate_element(modified_cif,
                                                        self.namespace,
                                                        '_atom_site_aniso_label',
                                                        element.group()))
                isolated_cifs[i] = modified_cif

            self.isolated_cifs = isolated_cifs
            self.elements = elements

            if output_path:
                self.write_to_file(isolated_cifs, elements, output_path + self.icsd_code + "/")

        # Remove all the positively charged elements from the cif file
        elif self.isolate == "anions":
            modified_cif = ReadCif(filepath)
            cations = []
            anions = []
            for element in elements_charge:
                if element.find("-") > 0:
                    anions.append(element)

                elif element.find("+") > 0:
                    cations.append(element)

            if '_atom_site_label' in cif:
                    isolated_cifs = (self.isolate_elements(modified_cif,
                                                        self.namespace,
                                                        '_atom_site_label',
                                                        anions))
            if '_atom_site_aniso_label' in cif:
                    isolated_cifs = (self.isolate_elements(modified_cif,
                                                        self.namespace,
                                                        '_atom_site_aniso_label',
                                                        anions))
            print()

            if output_path:
                self.write_to_file(isolated_cifs, ["anion"], output_path + self.icsd_code + "/")

        print()

    def isolate_element(self, cif, namespace, loopLabel, searchTerm):
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


    # Similar to above but takes a list instead
    def isolate_elements(self, cif, namespace, loopLabel, searchTerms):
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
            for element in searchTerms:
                if atom_no_numeric in element:
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
                if self.isolate == "elements":
                    filename = self.icsd_code + "_" + element.group() + ".cif"
                    outfile = open(output_path + filename, "w")
                    outfile.write(cifs[i].WriteOut())
                else:
                    filename = self.icsd_code + "_" + element + ".cif"
                    outfile = open(output_path + filename, "w")
                    outfile.write(cifs.WriteOut())
                # x = cifs[i].WriteOut()
                outfile.close()

if __name__ == "__main__":
    testpath = '/home/cameron/Datasets/ICSD/ICSD_2019_Li_CIFs/icsd_004102.cif'
    output_path = '/home/cameron/Documents/tmp/icsd_004102/'
    x = ElementIsolator(testpath, output_path)

    """
    Below function for a very bizarre reason isn't working although pretty much
    identical to above. Left here for helping track down the issue raised in
    https://bitbucket.org/jamesrhester/pycifrw/issues/18/writeout-str-representation-appears-to-be

    def isolate_element(self, element, cif, atom_site):
        # get the positions of each of the atoms and indices of our
        # specific element, ignores charge on the ion
        atomSiteLabel = cif[self.namespace][atom_site]
        elementIndices = [i for i, x in enumerate(atomSiteLabel)
                                        if x.find(element) > -1]

        # If element in the formula but not in atomic sites, exit
        if not elementIndices:
            return -1

        # Make a copy of the values of the specific element and remove this from
        # the Cif file. This is roundabout, but crashes if you do it directly as
        # you are accessing a list which you are also removing items from
        featureList = []
        keys = list(cif[self.namespace].GetLoop(atom_site).keys()[:])
        counter = 0

        for key in keys:
            featureList.append(cif[self.namespace][key][i] for i in elementIndices)
            cif[self.namespace].RemoveItem(key)

        if elementIndices:
            cif[self.namespace].AddLoopItem((keys, featureList))
            print(cif)

        # for i, key in enumerate(keys):
        #     cif.AddItem(key, featureList[i])

        # Formatting is very bugged once loop is created, issue created on the
        # repository https://bitbucket.org/jamesrhester/pycifrw/issues/18/writeout-str-representation-appears-to-be

        # returnCif = CifFile()
        # returnCif[self.namespace] = cif
        # returnCif[self.namespace].CreateLoop(keys)

        return cif
    """
