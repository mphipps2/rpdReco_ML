import os
import uproot3
import pandas as pd
import numpy as np
import time
import sys


def get_charge(df):
    #get the charge and discrad the total charge
    return df.iloc[:, df.columns.str.contains('Charge')].iloc[:,:-1]



def get_df(path, treeName = 'tree', branch = []):
    file = uproot3.open(path)
    #   print("file: ", file[treeName])
    tree = file[treeName]
    #    tree.show()
    #   print("branch: " , branch)
#    nNeutrons = len(tree.array(branch[0]))
#    print("nNeutrons: ", nNeutrons)
    if not branch:
        branch = tree.keys()
#    print("branch: " , branch)

#    tmp = tree.pandas.df(branch , flatten=False)
    df = tree.pandas.df(branch , entrystart=-1, flatten=False)
    return df


def combine_file(pos_path,eventNumber):
    try:
        particle = get_df(pos_path, treeName= "Particle", branch = ["nNeutrons","avgX","avgY","RP_true_value","RP_gen_value","pt_nuclear"])
        analysis = get_df(pos_path, treeName = "AnalysisTree", branch = ["rpd*Peak_max"])
    except FileNotFoundError:
        return None
    except KeyError:
        return None

    particleIndex = pd.Index([eventNumber],name="Event")
    particle.set_index(particleIndex,inplace=True,drop=True)

    analysisIndex = pd.Index([eventNumber],name="Event")
    analysis.set_index(analysisIndex,inplace=True,drop=True)

    out = pd.concat([particle, analysis], axis=1, sort=False)
    return out




def combine_file_in_folder(event_start, event_end):
    output = None
    start = time.time()
    folder_path = "../data/ToyFermi_qqFibers_LHC_noPedNoise/root/"
    err = 0
    err_log = []

    #for i in range(1000000):
    for i in range(event_start, event_end + 1):

        file = f"event{i}{side}.root"

        path = folder_path+file
        tmp = combine_file(path,i)
        if tmp is None:
            err += 1
            print("number of err file: ",err)
            err_log.append(i)
            continue
        if output is None:
            output = tmp

        else:
            output = output.append(tmp)

        #if i % 1000 == 0 and i != event_start:
        if i == event_end:
            print(output)
#            output.to_pickle(f"../data/ToyFermi_qqFIbers_LHC_noPedNoise/{side}_{i}.pickle")
#            output.to_pickle(f"/home/mike/Desktop/rpdReco/reco/data/ToyFermi_qqFibers_LHC_noPedNoise/{side}_{i}.pickle")
            output.to_pickle(f"../data/ToyFermi_qqFibers_LHC_noPedNoise/{side}_{i}.pickle")
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
