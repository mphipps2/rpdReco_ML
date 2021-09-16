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




if __name__ == '__main__':
        debug = False
        model_loss = "mse"
        model_num = 10
        model_type = "cnn_test"
        use_neutrons = False
        do_com = True
        do_z_norm = False
        do_model_residuals = True
        model_type_2 = "fcn"
        model_num_2 = 100
        model_loss_2 = "mse"
        do_z_norm_2 = False
        use_neutrons_2 = False
        do_ensemble_avg = True
        scenario = "ToyFermi_qRods_LHC_noPedNoise/"
        data_path = "../data/"+scenario
        model_path = "../models/"+scenario
        data_file = "test_A.pickle"
        data_znorm = "test_A_znorm.npy"
        output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Testing/" + scenario + "Model" + str(model_num) + "/"
        upperRange_gen = 1.4
        upperRange_truth = 2.5
        upperRange_predictionRes = 0.5
        # vertical offset in orginal data
        centerX_bias = 0
        centerY_bias = -0.471659

        #load trained model
        if debug:
                print("loading model")
        model = keras.models.load_model(model_path + f'model{model_type}_{model_num}_{model_loss}.h5',compile = False)

        #load dataset
        if debug:
                print("loading data")
        
        test_A = pd.read_pickle(data_path + data_file).to_numpy()

	#set test_x based on model: 8:24 for allchan, 24:32 for avg, 24:26 for CoM
        test_X_A = test_A[:,8:24]
        if model_type == 'cnn' or model_type == 'cnn_test':
                #reshape from (None, 16) to (None, 6, 6, 1)
                test_X_A = process.reshape_signal(test_X_A)
        
        Q_avg_A = test_A[:,0:2]
        psi_truth_A = test_A[:,5]
        pt_nuc_A = test_A[:,4] * 1000
        neutrons_A = test_A[:,7]
        neutrons_A_smeared = process.blur_neutron(test_A[:,7])        
        print ('real # of neutrons: ', neutrons_A, ' smeared #: ', neutrons_A_smeared)
        if do_z_norm:
                rpdSignals_A = np.load(data_path + data_znorm)
        else:
                rpdSignals_A = test_A[:,8:24]

        psi_gen_A = np.arctan2(Q_avg_A[:,1],Q_avg_A[:,0])
        if use_neutrons:
                Q_predicted_A = model.predict([neutrons_A_smeared.astype('float'), test_X_A.astype('float')])
        else:
                Q_predicted_A = model.predict([test_X_A.astype('float')])
        psi_rec_A = np.arctan2(Q_predicted_A[:,1],Q_predicted_A[:,0])
        psi_gen_rec_A = process.GetPsiResidual_np(psi_gen_A, psi_rec_A)
        psi_truth_rec_A = process.GetPsiResidual_np(psi_truth_A, psi_rec_A)

        if do_com:
                com_A = process.findCOM_np(rpdSignals_A)
                psi_com_A = process.getCOMReactionPlane_np(com_A, centerX_bias, centerY_bias)
                psi_gen_com_A = process.GetPsiResidual_np(psi_gen_A, psi_com_A)
                psi_truth_com_A = process.GetPsiResidual_np(psi_truth_A, psi_com_A)


        if do_model_residuals:
                model_2 = keras.models.load_model(model_path + f'model{model_type_2}_{model_num_2}_{model_loss_2}.h5',compile = False)
                test_X_A_2 = test_A[:,8:24]
                if model_type_2 == 'cnn' or model_type_2 == 'cnn_test':
                        #reshape from (None, 16) to (None, 6, 6, 1)
                        test_X_A_2 = process.reshape_signal(test_X_A_2)

                if do_z_norm_2:
                        rpdSignals_A_2 = np.load(data_path + data_znorm)
                else:
                        rpdSignals_A_2 = test_A[:,8:24]

                psi_gen_A_2 = np.arctan2(Q_avg_A[:,1],Q_avg_A[:,0])
                if use_neutrons_2:
                        Q_predicted_A_2 = model_2.predict([neutrons_A_smeared.astype('float'), test_X_A_2.astype('float')])
                else:
                        Q_predicted_A_2 = model_2.predict([test_X_A_2.astype('float')])
                psi_rec_A_2 = np.arctan2(Q_predicted_A_2[:,1],Q_predicted_A_2[:,0])
                psi_res_model1_model2 = process.GetPsiResidual_np(psi_rec_A, psi_rec_A_2)
                print("~~~ psi_res_model1_model2.dtype " , psi_res_model1_model2.dtype, " psi+rec_A " , psi_rec_A.dtype , " psi_rec_A_2 " , psi_rec_A_2.dtype, " psi_gen_rec_A " , psi_gen_rec_A.dtype, " psi_gen_A " , psi_gen_A.dtype, " psi_rec_A " , psi_rec_A.dtype)

        if do_ensemble_avg:
                Q_ensemble_A = np.mean( np.array([Q_predicted_A,Q_predicted_A_2]),axis=0)
                print("Q_CNN: " , Q_predicted_A)
                print("Q_FCN: " , Q_predicted_A_2)
                print("Q_ensemble: " , Q_ensemble_A)
                psi_ensemble_A = np.arctan2(Q_ensemble_A[:,1],Q_ensemble_A[:,0])
                psi_gen_ensemble_A = process.GetPsiResidual_np(psi_gen_A, psi_ensemble_A)
                psi_truth_ensemble_A = process.GetPsiResidual_np(psi_truth_A, psi_ensemble_A)
                
        Path(output_path).mkdir(parents=True, exist_ok=True)
        # integrated residuals (not broken up between # of neutrons and pt_nuc)
        vis_root.PlotResiduals(psi_gen_rec_A, "psi_gen_rec", output_path)
        vis_root.PlotResiduals(psi_truth_rec_A, "psi_truth_rec", output_path)
        vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_rec_A, upperRange_gen, "psi_gen_rec", output_path, save_residuals = False)
        vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_rec_A, upperRange_truth, "psi_truth_rec", output_path, save_residuals = False)
        
        if do_com:
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_com_A, upperRange_gen, "psi_gen_com", output_path, save_residuals = False)
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_com_A, upperRange_truth, "psi_truth_com", output_path,save_residuals = False)

        if do_model_residuals:
                vis_root.PlotPredictionResiduals(psi_res_model1_model2, model_type, model_type_2, "psi_mod1_mod2", output_path)
                vis_root.PlotPredictionResiduals_neutron(neutrons_A, pt_nuc_A, psi_res_model1_model2, model_type, model_type_2, upperRange_predictionRes, "psi_mod1_mod2", output_path, save_residuals = False)

        if do_ensemble_avg:
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_ensemble_A, upperRange_gen, "psi_gen_ensemble", output_path, save_residuals = False)
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_ensemble_A, upperRange_truth, "psi_truth_ensemble", output_path, save_residuals = False)



	
