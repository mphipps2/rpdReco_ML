import os
import uproot
import pandas as pd
import numpy as np
import time
import sys


def get_charge(df):
    #get the charge and discrad the total charge
    return df.iloc[:, df.columns.str.contains('Charge')].iloc[:,:-1]



def get_df(path, tree = 'tree', branch = []):
    file = uproot.open(path)
    if not branch:
        branch = file[tree].keys()
    df = file[tree].pandas.df(branch , flatten=False)
    return df


def combine_file(pos_path):
    try:
        event_data = get_df(pos_path, tree = "EventData")
        analysis = get_df(pos_path, tree = "AnalysisTree")
    except FileNotFoundError:
        return None
    except KeyError:
        return None
    charge = get_charge(analysis)
    event_data['Event number'] = int(pos_path.split("_")[-1][:-12])
    out = pd.concat([event_data, charge], axis=1, sort=False)
    return out




def combine_file_in_folder(event_start, event_end):
    output = None
    start = time.time()
    folder_path = "/projects/engrit/jzcapa/Users/Sheng/ToyV1_Fermi_single_pT_2.7TeV_Merge_Charge_030621/"
    err = 0
    err_log = []

    #for i in range(1000000):
    for i in range(event_start, event_end + 1):

        file = f"Merge_{side}_output_Toy_Fermi_{i}output1.root"

        path = folder_path+file
        tmp = combine_file(path)
        if tmp is None:
            err += 1
            print("number of err file: ",err)
            err_log.append(i)
            continue
        if output is None:
            output = tmp

        else:
            output = output.append(tmp)


        if i % 1000 == 0 and i != event_start:

            output.to_pickle(f"./tmp/{side}_{i}.pickle")
            output = None
            print(i,  time.time() - start)
            start = time.time()
        #output = output.to_numpy().astype(float)
        #np.save(f"./Output/RPD_signal/Root_charge_signal080820/result{i}.npy", output)
    #output.to_pickle(f"./tmp/{side}_{i}.pickle")
    with open(f'./log/{side}_{event_start}log.txt', 'a+') as logfile:
        print(err_log)
        for i in err_log:
            logfile.write(str(i) + "\n")
    print(err)
if __name__ == '__main__':
    #print(sys.argv)

    print("Program starts")
    side = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    combine_file_in_folder(start, end)
#combine_file_in_folder("/projects/engrit/jzcapa/Users/Sheng/ToyV1_Fermi_2.7TeV_Merge_Charge_110920/")
