from pathlib import Path

# Add package to python path in .bashrc file: eg) export PYTHONPATH='/home/mike/Desktop/rpdReco/reco/'
import lib.norm as norm
import lib.models as models
import lib.process as process
import lib.vis_root as vis_root
import lib.vis_mpl as vis_mpl
import lib.io as io

import numpy as np
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import pandas as pd
from array import array
import time



if __name__ == '__main__':
        debug = False
        model_loss = "mse"
#        model_num = 200
        model_num = 1
        model_type = "cnn"
        model_type_1_label = "cnn_qqFibers"
#        model_type_1_label = "fcn_qqFibers"
#        model_type_1_label = "linear_qqFibers"
#        model_type_1_label = "cnn_qpFibers"
#        model_type_1_label = "cnn_mini_qpFibers"
        com_label = "com_qqFibers"
#        com_label = "com_qpFibers"
        use_neutrons = False
        do_com = False
        do_z_norm = False
        do_small_batch = False
        use_unit_vector = True
        use_padding = True
        two_trainer_ratio = 0.6
        two_trainer_filename = "40batch"
        do_model_residuals = False
        model_type_2 = "cnn"
#        model_type_2_label = "fcn_qqFibers"
#        model_type_2_label = "cnn_qqFibers"
        model_type_2_label = "cnn_mini_qpFibers"
        model_num_2 = 1
        model_loss_2 = "mse"
        model_2_is_qRod = False
        model_2_is_qq = True
        do_position_resolution = False
        do_truth_pos_plot = False
        do_subtracted_channel_plot = False
        do_z_norm_2 = False
        use_neutrons_2 = False
        do_ensemble_avg = False
        do_ratio_plot_ptnuc = False
        scenario = "ToyFermi_qqFibers_LHC_noPedNoise/"
#        scenario = "ToyFermi_qpFibers_LHC_noPedNoise/"
 #       scenario = "ToyFermi_mini_qpFibers_LHC_noPedNoise/"
#        scenario_2 = "ToyFermi_qqFibers_LHC_noPedNoise/"
        scenario_2 = "ToyFermi_mini_qpFibers_LHC_noPedNoise/"
#        scenario_2 = "ToyFermi_qpFibers_LHC_noPedNoise/"
        data_path = "../data/"+scenario
        model_path = "../models/"+scenario
        data_path_2 = "../data/"+scenario_2
        model_path_2 = "../models/"+scenario_2
#        data_file = "test_A_25k.pickle"
        data_file_2 = "test_A_200k.pickle"
#        data_file = "test_A.pickle"
        data_file = "test_A_1k.pickle"
#        data_file = "test_A.pickle"
#        data_file = "test_A_50k.pickle"
        data_znorm = "test_A_znorm.npy"
        output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Testing/" + scenario + "Model" + str(model_num) + "/"
        upperRange_gen = 0.95
        upperRange_truth = 2.5
        upperRange_predictionRes = 0.5
        # vertical offset in orginal data
        centerX_bias = 0
        # for qq
        #        centerY_bias = 0.67223859
        # for qp
#        centerY_bias = 0.367
        # for mini_qp
        centerY_bias = 0.272
        #load trained model
        if debug:
                print("loading model")

        # if we are ensembling we train the models with less data
        if do_small_batch:
                model = keras.models.load_model(model_path + f'model{model_type}_{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.h5',compile = False)
        else:
                model = keras.models.load_model(model_path + f'model{model_type}_{model_num}_{model_loss}.h5',compile = False)
                
        #load dataset
        if debug:
                print("loading data")
        
        test_A = pd.read_pickle(data_path + data_file).to_numpy()

	#set test_x based on model: 6:22 for allchan, 24:32 for avg, 24:26 for CoM
        test_X_A = test_A[:,6:22]
                
        if model_type == 'cnn' or model_type == 'cnn_test':
        #reshape from (None, 16) to (None, 6, 6, 1)
                if use_padding: 
                        test_X_A = process.reshape_signal(test_X_A)
                else:
                        test_X_A = process.reshape_signal(test_X_A,normalization=False, flatten=False, padding = False)

        Q_avg_A = test_A[:,1:3]

        if use_unit_vector:
                Q_avg_A = norm.get_unit_vector(Q_avg_A)

        psi_truth_A = test_A[:,3]
        pt_nuc_A = test_A[:,5] * 1000
        neutrons_A = test_A[:,0]
        neutrons_A_smeared = process.blur_neutron(test_A[:,0])        
        print ('real # of neutrons: ', neutrons_A, ' smeared #: ', neutrons_A_smeared)
        if do_z_norm:
                rpdSignals_A = np.load(data_path + data_znorm)
        else:
                rpdSignals_A = test_A[:,6:22]

        psi_gen_A = np.arctan2(Q_avg_A[:,1],Q_avg_A[:,0])

#        t0 = time.clock()
        t0 = time.time()
        if use_neutrons:
                Q_predicted_A = model.predict([neutrons_A_smeared.astype('float'), test_X_A.astype('float')])
        else:
                Q_predicted_A = model.predict([test_X_A.astype('float')])
        print("training took: ", time.time() - t0, " seconds")
        
#        print("model2: residual: " , psi_gen_res_model2)
    
        psi_rec_A = np.arctan2(Q_predicted_A[:,1],Q_predicted_A[:,0])
        print("here: psi_rec_A " , psi_rec_A)
