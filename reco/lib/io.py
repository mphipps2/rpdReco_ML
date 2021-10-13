import os
import lib.process as process
import time
import pandas as pd
import uproot3
import sys

def get_dataset(filename = "./Data/reduced_tree_tmp/Acharge.pickle"):
        output = []
        start = time.time()

        data = pd.read_pickle(filename)
        data = data.drop_duplicates()

#        if subtract:
#                data = process.subtract_signals(data)
                
        return data

def get_dataset_peak(filename = "./Data/reduced_tree_tmp/Acharge.pickle"):
        output = []
        start = time.time()

        data = pd.read_pickle(filename)
        print("data columns: ", data.columns)
        print("data type: ", type(data))
#        data = data.drop_duplicates()
                
        return data

def get_dataset_peak_folder(foldername = "../Data/ToyFermi_qqFibers_LHC_noPedNoise/A_side/",side="A", subtract = True):
        output = []
        start = time.time()
        start_event = 9999
        end_event = 1000000
        increment = 10000
        merged_file = combine_files(start_event, end_event, increment, foldername, side)
        print(f"merged_file_{side}: ", merged_file)
#        print("data columns: ", merged_file.columns)
        print("data type: ", type(merged_file))
        data = merged_file.drop_duplicates()

        if subtract:
                data = process.subtract_signals_peak(data)
                
        return data

def combine_files(start,end,increment,folder = "./Data/Merged_charge_122420/", side = 'A'):
        output = []
        start_time = time.time()
        for i in range(start, end, increment):
                if i % 999999 == 0:
                        print("event", i, "time", time.time() - start_time)
                        start_time = time.time()
                print("reading file: ",f"{side}_{i}.pickle")
                output.append(pd.read_pickle(folder + f"{side}_{i}.pickle"))
        data = pd.concat(output).astype(float)
        print("data: ", data)
        return data
