'''
A Glue script to use CifToPers to process a folder of cif files in one batch

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

Usage: ./PersistenceGenerator.py [-r] INPUT_PATH OUTPUT_PATH (persistence VM on pc)

This class takes an input cif file, performs an expansion in x, y, z dimensions
and then saves a persistence diagram to a specified location in the pickle format
if we have no change in number of points on expansion in each given dimension.

Can be run using mpiexec if this is installed on your system in which case
specify the number of processors to spawn after -n and the command is:

mpiexec -n 8 python PersistenceGenerator.py -r "/INPUT/FOLDER" "/OUTPUT/FOLDER"

On my system this ate through memory (20GB) very quickly leading to kernel panic, to
get around this I spawn a new process and kill it if it takes more than 10 seconds
to process persistence, then I come back to these and do them unthreaded

mpiexec -n 4 python PersistenceGenerator.py -r -i '/home/cameron/Datasets/ICSD/MineralClass/InputCifFolder/' -o '/home/cameron/Datasets/ICSD/MineralClass/MineralPers/'
mpiexec -n 1 python PersistenceGenerator.py -i '/home/cameron/Datasets/ICSD/test_files/icsd_000373.cif' -o '/home/cameron/Documents/tmp/PersGen' -s "elements"

Simple command to run output to a text file
python PersistenceGenerator.py -i '/home/cameron/Datasets/ICSD/test_files/icsd_000373.cif' > out.txt


Options:
-r  Recursively search through the folder and output to a folder
-i Select input folder/file
-o select output folder
-t Select timeout when processing persistence. Can be useful for very computationally heavy files
-s splitting type, whether to split input cif into elements, ions, or no splitting at all
-l log files folder, default /var/logs/PersGen.log
TODO: Refactor, this code has become pretty messy with clumsy feature creep

'''
import argparse
import os
import sys
import time
import multiprocessing
import tempfile

import numpy as np

from mpi4py import MPI

from Percifter.ElementIsolator import ElementIsolator
from Percifter.IonIsolator import IonIsolator
from Percifter.CifToPers import CifToPers

def parse_arguments():
    parser = argparse.ArgumentParser(description='Recursively search through a folder and take the persistence of each cif file by first isolating each element, calculating the persistence of these, then writing these to file as a pickled object')
    parser.add_argument('-i', '--inputpath', default='./', help='The folder/file to process')
    parser.add_argument('-o', '--outputpath', default='./', help='The output path to write the file to')
    parser.add_argument('-r', '--recursive', action='store_true', help='Whether to recursively check folder')
    parser.add_argument('-t', '--timeout', default=1000, type=int, help='Length of time to process a single file before terminating process, should be increased for large files but may cause memory overflow if multiple large files are processed simultaneously')
    parser.add_argument('-s', '--splitting', default='ion', type=str, help='How to split up the input cif file for separate persistence diagrams. Available: "ion", "element", "none"')
    parser.add_argument('-l', '--logfile', default='/var/logs/PersGen.log', type=str, help="Where to save the logs for any files that may not have successfully processed")
    args = (vars(parser.parse_args()))
    return args

def get_paths(input_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # If using MPI
    num_processes = comm.size

    if rank == 0:                # First rank splits the files up
        filepaths = os.listdir(input_path)
        print("Building index...")
        indexes = np.arange(len(filepaths))
        np.random.shuffle(indexes)
        splits = np.array_split(indexes, num_processes)
    else:                      # All other processes
        filepaths = []
        splits = []

    # wait for process 0 to get the filepaths/splits and broadcast these
    comm.Barrier()
    filepaths = comm.bcast(filepaths, root=0)
    splits = comm.bcast(splits, root=0)

    # take only filepaths for our rank
    my_filepaths = [filepaths[x] for x in splits[comm.rank]]
    return my_filepaths

def process_cif(input_path, output_path, timeout, filename=None):
    filename_args = {'filename': filename}
    time_flag = True
    if filename:
        p = multiprocessing.Process(target=CifToPers,
            name="CifToPers",
            args=(input_path,
                    output_path),
            kwargs=filename_args)
    else:
        p = multiprocessing.Process(target=CifToPers,
            name="CifToPers",
            args=(input_path,
                    output_path))
    p.start()
    p_t_start = time.time()

    while p.is_alive() and time_flag:
        if time.time() - p_t_start < timeout:
            time.sleep(0.1)
        else:
            print("Out of time")
            time_flag = False

    if time_flag == False:
        with open(args['logfile'], "a+") as myfile:
                myfile.write(f"{input_path}\n")
                myfile.close()
    p.terminate()
    p.join()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # If using MPI
    num_processes = comm.size
    args = parse_arguments()

    input_path = args['inputpath']
    output_path = args['outputpath']
    # print(input_path)

    time_start = time.time()

    if args['recursive']:
        failed_cifs = []

        if os.path.isdir(input_path):    # if it is actually a folder
            my_filepaths = get_paths(input_path)

            my_files_processed = 0
            my_file_count = len(my_filepaths)
            print(f"Process {comm.rank} file count: {my_file_count}")

            for item in my_filepaths:
                # If these have been pre-split into isolated cifs then take the
                # folder, else take the cif file and split into cifs
                process_path = os.path.join(input_path, item)

                icsd_code = item if item[-4:] != '.cif' else item[:-4]
                this_out_dir = os.path.join(output_path, icsd_code)

                if not os.path.exists(this_out_dir):
                    os.makedirs(this_out_dir)

                if args['splitting'] == "ion":
                    # Here we assume that the given path points to a cif file
                    print("IONS")
                    isolated_cifs = IonIsolator(process_path)
                    elements = ["cations", "anions", "neutrals"]

                elif args['splitting'] == "elements":
                    isolated_cifs = ElementIsolator(process_path)
                    elements = isolated_cifs.elements
                    print(elements)

                print(isolated_cifs.isolated_cifs)

                for i, cif in enumerate(isolated_cifs.isolated_cifs):
                    if cif != None:
                        filename = icsd_code + "_" + elements[i] + '.pers'
                        pers_output_path = os.path.join(output_path, icsd_code, filename)

                        with tempfile.NamedTemporaryFile(mode="w+t") as tmp:
                            tmp.write(str(cif) + "\n")
                            tmp.flush()
                            process_cif(tmp.name, pers_output_path,
                                        args['timeout'], filename=filename)

                print(f"{icsd_code} processed")

            print("Completed from rank: " + str(rank))
            comm.Barrier()

            if rank == 0:
                print("Final time taken: " + str(time.time() - time_start)) #Final time taken: 39653.9169159

        else:
            sys.exit("Not a directory, remove -r flag")

    else:
        if os.path.isfile(input_path):
            filename = input_path.split('/')[-1]
            icsd_code = input_path.split('/')[-2]
            out_filename = filename[:-4] + ".pers"
            pers_output_path = os.path.join(output_path, icsd_code, out_filename)

            print(f"Processing {input_path}")
            process_cif(input_path, pers_output_path, args['timeout'],
                        filename=filename)
