'''
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

Generator takes input path and output path as arguments

Options:
-r  Recursively search through the folder and output to a folder

TODO
- Expand the single file manipulator to take in multi element cif files

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
from Percifter.CifToPers import CifToPers

def parse_arguments():
    parser = argparse.ArgumentParser(description='Recursively search through a folder and take the persistence of each cif file by first isolating each element, calculating the persistence of these, then writing these to file as a pickled object')
    parser.add_argument('-i', '--inputpath', default='./', help='The folder/file to process')
    parser.add_argument('-o', '--outputpath', default='./', help='The output path to write the file to')
    parser.add_argument('-r', '--recursive', action='store_true', help='Whether to recursively check folder')
    parser.add_argument('-t', '--timeout', default=1000, type=int, help='Length of time to process a single file before terminating process, should be increased for large files but may cause memory overflow if multiple large files are processed simultaneously')
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
        with open("/home/cameron/Datasets/ICSD/MineralClass/out_of_time_cif_1000", "a+") as myfile:
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
            # Remove below lines, for testing only
            # processed_paths = os.listdir('/home/cameron/Datasets/ICSD/MineralClass/MineralPers')
            # print(my_filepaths[0])
            # print(processed_paths[0])
            # print(len(my_filepaths))
            my_filepaths = [x for x in my_filepaths if x not in processed_paths]
            # print(len(my_filepaths))

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

                if os.path.isdir(process_path):
                    # Here we assume that this is a directory of isolated cifs
                    for cif in os.listdir(process_path):
                        cif_path = os.path.join(process_path, cif)

                        element = cif.split('_')[2].split('.')[0]
                        filename = icsd_code + "_" + element + '.pers'
                        pers_output_path = os.path.join(this_out_dir, filename)

                        process_cif(cif_path, pers_output_path, args['timeout'])

                else:
                    # Here we assume that the given path points to a cif file
                    isolated_cifs = ElementIsolator(process_path)
                    elements = isolated_cifs.elements
                    for i, cif in enumerate(isolated_cifs.isolated_cifs):
                        filename = icsd_code + "_" + elements[i].group() + '.pers'
                        pers_output_path = os.path.join(output_path, filename)

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
            filename = filename[:-4] + ".pers"

            output_folder = '/home/cameron/Datasets/ICSD/MineralClass/MineralPers/'
            pers_output_path = os.path.join(output_folder, icsd_code, filename)

            print(f"Processing {input_path}")
            process_cif(input_path, pers_output_path, args['timeout'],
                        filename=filename)


    # else:
    #     x = CifToPers(input_path, output_path)
    #     print("Persistence file written with expansion {} and {} points".format(x.xyz_expansion, x.pers['num_edges']))
