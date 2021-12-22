from pathlib import Path

# Add package to python path in .bashrc file: eg) export PYTHONPATH='/home/mike/Desktop/rpdReco/reco/'
import lib.norm as norm
import lib.models as models
import lib.process as process
import lib.vis_root as vis_root
import lib.vis_mpl as vis_mpl
import lib.io as io

from ROOT import *
import numpy as np
import pandas as pd
import numpy as np
import tensorflow.keras as keras
import pandas as pd
from array import array
import time
import math


if __name__ == '__main__':
        debug = False
        model_loss = "mse"
        model_num = 20
        model_type = "cnn"
#        model_type_1_label = "cnn_array_qpFibers"
        model_type_1_label = "cnn_qqFibers"
#        model_type_1_label = "qq_fibers"
#        model_type_1_label = "cnn_mini_qpFibers"
        com_label = "com_qqFibers"
#        com_label = "com_qpFibers"
        use_neutrons = False
        do_com = False
        do_z_norm = False
        do_small_batch = False
        do_2rpds = False
        use_unit_vector = True
        use_padding = True
        two_trainer_ratio = 0.6
        two_trainer_filename = "40batch"
        do_model_residuals = False
        model_type_2 = "cnn"
#        model_type_2_label = "fcn_qqFibers"
#        model_type_2_label = "cnn_qqFibers"
        model_type_2_label = "qp_fibers"
        model_num_2 = 2
        model_loss_2 = "mse"
        model_2_is_qRod = False
        model_2_is_qq = True
        do_position_resolution = False
        do_truth_pos_plot = False
        do_reco_prediction_plot = True
        do_subtracted_channel_plot = False
        do_neutron_dep = False
        do_pt_nuc_dep = False
        do_z_norm_2 = False
        use_neutrons_2 = False
        do_ensemble_avg = False
        do_ratio_plot_ptnuc = False
        do_AC_res = True
        do_C_side_res = False
        do_q_res = False
#        scenario = "ToyFermi_array_qpFibers_LHC_noPedNoise/"
#        scenario = "ToyFermi_qpFibers_LHC_noPedNoise/"
 #       scenario = "ToyFermi_mini_qpFibers_LHC_noPedNoise/"
        scenario = "ToyFermi_qqFibers_LHC_noPedNoise/"
        scenario_2 = "ToyFermi_qqFibers_LHC_noPedNoise/"
#        scenario_2 = "ToyFermi_array_qpFibers_LHC_noPedNoise/"
#        scenario_2 = "ToyFermi_qpFibers_LHC_noPedNoise/"
        data_path = "../data/"+scenario
        model_path = "../models/"+scenario
        data_path_2 = "../data/"+scenario_2
        model_path_2 = "../models/"+scenario_2
#        data_file = "test_A_25k.pickle"
        data_file_2 = "test_A_100k_rpd1.pickle"
#        data_file_A = "test_A_100k.pickle"
#        data_file_A = "test_A_100k_rpd1.pickle"
        data_file_A = "test_A_100k.pickle"
        data_file_B = "test_B_100k.pickle"
        data_file_2A = "test_A_100k_rpd2.pickle"
        data_file_2B = "test_B_100k_rpd2.pickle"
#        data_file = "test_A.pickle"
#        data_file = "test_A_50k.pickle"
        data_znorm_A = "test_znorm_A.npy"
        data_znorm_B = "test_znorm_B.npy"
        output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Testing/" + scenario + "Model" + str(model_num) + "/"
        upperRange_gen = 0.95
        upperRange_truth = 2.5
        upperRange_predictionRes = 0.5
        # vertical offset in orginal data
        centerX_bias = 0
        # for qq
        centerY_bias = 0.67223859
        # for qp
#        centerY_bias = 0.367
        # for mini_qp
#        centerY_bias = 0.272
        #load trained model

        print("loading model")

        # if we are ensembling we train the models with less data
        if do_small_batch:
                model = keras.models.load_model(model_path + f'model{model_type}_{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.h5',compile = False)
        else:
                model = keras.models.load_model(model_path + f'model{model_type}_{model_num}_{model_loss}.h5',compile = False)

        if do_2rpds:
                model2 = keras.models.load_model(model_path_2 + f'model{model_type_2}_{model_num_2}_{model_loss_2}.h5',compile = False)       
        #load dataset
        if debug:
                print("loading data")
        
        test_A = pd.read_pickle(data_path + data_file_A).to_numpy()
        test_B = pd.read_pickle(data_path + data_file_B).to_numpy()

	#set test_x based on model: 6:22 for allchan, 24:32 for avg, 24:26 for CoM
        test_X_A = test_A[:,6:22]
        test_X_B = test_B[:,6:22]
                
        if model_type == 'cnn' or model_type == 'cnn_test':
        #reshape from (None, 16) to (None, 6, 6, 1)
                if use_padding: 
                        test_X_A = process.reshape_signal(test_X_A)
                        test_X_B = process.reshape_signal(test_X_B)
                else:
                        test_X_A = process.reshape_signal(test_X_A,normalization=False, flatten=False, padding = False)
                        test_X_B = process.reshape_signal(test_X_B,normalization=False, flatten=False, padding = False)

        Q_avg_A = test_A[:,1:3]
        Q_avg_B = test_B[:,1:3]
        Q_avg_x_A = Q_avg_A[:,0]
        Q_avg_y_A = Q_avg_A[:,1]

        Qx_A = Q_avg_A[:,0]
        Qy_A = Q_avg_A[:,1]

        if use_unit_vector:
                Q_avg_A = norm.get_unit_vector(Q_avg_A)
                Q_avg_B = norm.get_unit_vector(Q_avg_B)

        psi_truth_A = test_A[:,3]
        pt_nuc_A = test_A[:,5] * 1000
        neutrons_A = test_A[:,0]
        neutrons_A_smeared = process.blur_neutron(test_A[:,0])
        # note: default method in v1Gen is that psi_truth_A == psi_truth_B
        psi_truth_B = test_B[:,3]

        pt_nuc_B = test_B[:,5] * 1000
        neutrons_B = test_B[:,0]        
        neutrons_B_smeared = process.blur_neutron(test_B[:,0])
        neutrons_AC = (neutrons_A + neutrons_B) / 2
        print ('real # of A-side neutrons: ', neutrons_A, ' smeared #: ', neutrons_A_smeared)
        print ('real # of B-side neutrons: ', neutrons_B, ' smeared #: ', neutrons_B_smeared)
        if do_z_norm:
                rpdSignals_A = np.load(data_path + data_znorm_A)
                rpdSignals_B = np.load(data_path + data_znorm_B)
        else:
                rpdSignals_A = test_A[:,6:22]
                rpdSignals_B = test_B[:,6:22]

        psi_gen_A = np.arctan2(Q_avg_A[:,1],Q_avg_A[:,0])
        psi_gen_B = np.arctan2(Q_avg_B[:,1],Q_avg_B[:,0])
        # flip gen vec so we're looking at it from same perspective as A side
        psi_gen_B = np.where(psi_gen_B > 0, psi_gen_B -  math.pi, psi_gen_B + math.pi)

        if use_neutrons:
                Q_predicted_A = model.predict([neutrons_A_smeared.astype('float'), test_X_A.astype('float')])
                Q_predicted_B = model.predict([neutrons_B_smeared.astype('float'), test_X_B.astype('float')])
        else:
                Q_predicted_A = model.predict([test_X_A.astype('float')])
                Q_predicted_B = model.predict([test_X_B.astype('float')])
 
        print ("Qx: " , Qx_A)
        print ("Qy: " , Qy_A)
        print ("Qx_pred: " , Q_predicted_A[:,0])
        print ("Qy_pred: " , Q_predicted_A[:,1])
        if do_reco_prediction_plot:
                vis_root.PlotRecoPos(Q_predicted_A[:,0], Q_predicted_A[:,1], "reco_pred", output_path)
        
        psi_rec_A = np.arctan2(Q_predicted_A[:,1],Q_predicted_A[:,0])
 #       print("here: psi_rec_A " , psi_rec_A)
        psi_gen_rec_A = process.GetPsiResidual_np(psi_gen_A, psi_rec_A)
        psi_truth_rec_A = process.GetPsiResidual_np(psi_truth_A, psi_rec_A)

        qx_gen_rec_A_unit = process.GetPosResidual_np(Q_avg_A[:,0], Q_predicted_A[:,0])
        qy_gen_rec_A_unit = process.GetPosResidual_np(Q_avg_A[:,1], Q_predicted_A[:,1])

        qx_gen_rec_A = process.GetPosResidual_np(Qx_A, Q_predicted_A[:,0])
        qy_gen_rec_A = process.GetPosResidual_np(Qy_A, Q_predicted_A[:,1])
        q_mag_gen_rec_A = process.GetPosResidual_mag_np(Q_avg_A[:,0], Q_avg_A[:,1], Q_predicted_A[:,0], Q_predicted_A[:,1])
        com_A = process.findCOM_np(rpdSignals_A)
        psi_com_A = process.getCOMReactionPlane_np(com_A, centerX_bias, centerY_bias)        
        
#        print("B q_y " , Q_predicted_B[:,1] , " q_x " , Q_predicted_B[:,0])
        psi_rec_B = np.arctan2(Q_predicted_B[:,1],Q_predicted_B[:,0])
        # flip rec vec so we're looking at it from same perspective as A side
        psi_rec_B_flip = np.where(psi_rec_B > 0, psi_rec_B - math.pi, psi_rec_B + math.pi)



        if do_2rpds:
                test_2A = pd.read_pickle(data_path + data_file_2A).to_numpy()
                test_2B = pd.read_pickle(data_path + data_file_2B).to_numpy()
                test_X_2A = test_2A[:,6:22]
                test_X_2B = test_2B[:,6:22]
                test_X_2A = process.reshape_signal(test_X_2A)
                test_X_2B = process.reshape_signal(test_X_2B)
                rpdSignals_2A = test_2A[:,6:22]
                rpdSignals_2B = test_2B[:,6:22]
                Q_predicted_2A = model2.predict([test_X_2A.astype('float')])
                Q_predicted_2B = model2.predict([test_X_2B.astype('float')])
                psi_rec_2A = np.arctan2(Q_predicted_2A[:,1],Q_predicted_2A[:,0])
                psi_rec_2B = np.arctan2(Q_predicted_2B[:,1],Q_predicted_2B[:,0])
                psi_rec_2B_flip = np.where(psi_rec_2B > 0, psi_rec_2B - math.pi, psi_rec_2B + math.pi)
                psi_gen_rec_2A = process.GetPsiResidual_np(psi_gen_A, psi_rec_2A)
                psi_truth_rec_2A = process.GetPsiResidual_np(psi_truth_A, psi_rec_2A)
                
#        print("psi_gen_A " , psi_gen_A)
#        print("psi_gen_B " , psi_gen_B)
        if do_AC_res:
                epA_gen_x = np.cos(psi_gen_A)
                epA_gen_y = np.sin(psi_gen_A)
                epB_gen_x = -1*np.cos(psi_gen_B)
                epB_gen_y = -1*np.sin(psi_gen_B)
#                print("Ax " , epA_gen_x, " y " , epA_gen_y)
#                print("Bx " , epB_gen_x, " y " , epB_gen_y)
                epAB_gen_x = epA_gen_x + epB_gen_x
                epAB_gen_y = epA_gen_y + epB_gen_y
                psi_gen = np.arctan2(epAB_gen_y,epAB_gen_x)
                print("psi_gen: ", psi_gen, " psi_gen_A: ", psi_gen_A, " psi_gen_B " , psi_gen_B)
                
                epA_rec_x = np.cos(psi_rec_A)
                epA_rec_y = np.sin(psi_rec_A)
                epB_rec_x = -1*np.cos(psi_rec_B)
                epB_rec_y = -1*np.sin(psi_rec_B)                
                epAB_rec_x = epA_rec_x + epB_rec_x
                epAB_rec_y = epA_rec_y + epB_rec_y                
                psi_rec = np.arctan2(epAB_rec_y,epAB_rec_x)
                print("psi_rec: ", psi_rec, " psi_rec_A: ", psi_rec_A, " psi_rec_B " , psi_rec_B)

                psi_rec_diff_AB = psi_rec_B_flip - psi_rec_A
                print("psi_recA: " , psi_rec_A , " psi_rec_B " , psi_rec_B, " diff: " , psi_rec_diff_AB)
                sp_QVec_rec_AB = (Q_predicted_A[:,0] * (-1*Q_predicted_B[:,0])) + (Q_predicted_A[:,1] * (-1*Q_predicted_B[:,1]))
                psi_gen_rec_AB = process.GetPsiResidual_np(psi_gen, psi_rec)                
                psi_truth_rec_AB = process.GetPsiResidual_np(psi_truth_B, psi_rec)
                
                if do_2rpds:                        
                        epA_rec_2x = np.cos(psi_rec_2A)
                        epA_rec_2y = np.sin(psi_rec_2A)
                        epA_rec_2y = np.sin(psi_rec_2A)
                        epB_rec_2x = -1*np.cos(psi_rec_2B)
                        epB_rec_2y = -1*np.sin(psi_rec_2B)
                        epA_rec_1x2x = epA_rec_x + epA_rec_2x
                        epA_rec_1y2y = epA_rec_y + epA_rec_2y
                        epB_rec_1x2x = epB_rec_x + epB_rec_2x
                        epB_rec_1y2y = epB_rec_y + epB_rec_2y
                        epAB_rec_12x = epA_rec_x + epA_rec_2x + epB_rec_x + epB_rec_2x
                        epAB_rec_12y = epA_rec_y + epA_rec_2y + epB_rec_y + epB_rec_2y                        
                        psi_rec_12 = np.arctan2(epAB_rec_12y,epAB_rec_12x)

                        epAB_rec_2x = epA_rec_2x + epB_rec_2x
                        epAB_rec_2y = epA_rec_2y + epB_rec_2y
                        psi_rec_2 = np.arctan2(epAB_rec_2y,epAB_rec_2x)
                        
                        psi_recA_1xy_2xy = np.arctan2(epA_rec_1y2y,epA_rec_1x2x)
                        psi_recB_1xy_2xy = np.arctan2(epB_rec_1y2y,epB_rec_1x2x)
                        psi_rec_diff_2AB = psi_rec_B_flip - psi_rec_A                
                        psi_gen_rec_12AB = process.GetPsiResidual_np(psi_gen, psi_rec_12)                
                        psi_truth_rec_12AB = process.GetPsiResidual_np(psi_truth_B, psi_rec_12)
                        psi_gen_rec_2AB = process.GetPsiResidual_np(psi_gen, psi_rec_2)                
                        psi_truth_rec_2AB = process.GetPsiResidual_np(psi_truth_B, psi_rec_2)

                        psi_rec_diff_12_AB = psi_recB_1xy_2xy - psi_recA_1xy_2xy

                        
        if do_C_side_res:
                print(" psi_rec_A: ", psi_rec_A, " psi_rec_B " , psi_rec_B)
                print(" psi_gen_A: ", psi_gen_A, " psi_gen_B " , psi_gen_B)
                print(" psi_truth_A: ", psi_truth_A, " psi_truth_B " , psi_truth_B)
                psi_gen_rec_B = process.GetPsiResidual_np(psi_gen_B, psi_rec_B)                
                psi_truth_rec_B = process.GetPsiResidual_np(psi_truth_B, psi_rec_B_flip)

        if do_com:
                psi_gen_com_A = process.GetPsiResidual_np(psi_gen_A, psi_com_A)
                psi_truth_com_A = process.GetPsiResidual_np(psi_truth_A, psi_com_A)

                com_B = process.findCOM_np(rpdSignals_B)
                psi_com_B = process.getCOMReactionPlane_np(com_B, centerX_bias, centerY_bias)
                psi_com_B_flip = np.where(psi_com_B > 0, psi_com_B - math.pi, psi_com_B + math.pi)
                psi_gen_com_B = process.GetPsiResidual_np(psi_gen_B, psi_com_B)
                psi_truth_com_B = process.GetPsiResidual_np(psi_truth_B, psi_com_B_flip)


        if do_model_residuals:
                model_2 = keras.models.load_model(model_path + f'model{model_type_2}_{model_num_2}_{model_loss_2}.h5',compile = False)
                test_X_A_2 = test_A[:,6:22]
                if model_type_2 == 'cnn' or model_type_2 == 'cnn_test':
                        #reshape from (None, 16) to (None, 6, 6, 1)
                        test_X_A_2 = process.reshape_signal(test_X_A_2)

                if do_z_norm_2:
                        rpdSignals_A_2 = np.load(data_path + data_znorm)
                else:
                        rpdSignals_A_2 = test_A[:,6:22]

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
                psi_ensemble_A = np.arctan2(Q_ensemble_A[:,2],Q_ensemble_A[:,1])
                psi_gen_ensemble_A = process.GetPsiResidual_np(psi_gen_A, psi_ensemble_A)
                psi_truth_ensemble_A = process.GetPsiResidual_np(psi_truth_A, psi_ensemble_A)
                
        if do_ratio_plot_ptnuc:
                if (model_2_is_qRod):                        
                        model_2 = keras.models.load_model(model_path_2 + f'model{model_type_2}_{model_num_2}_{model_loss_2}.h5',compile = False)
                        test_A_2 = pd.read_pickle(data_path_2 + data_file_2).to_numpy()
                        test_X_A_2 = test_A_2[:,8:24]
                        pt_nuc_A_2 = test_A_2[:,4] * 1000
                        if do_z_norm_2:
                                rpdSignals_A_2 = np.load(data_path_2 + data_znorm)
                        else:
                                rpdSignals_A_2 = test_A_2[:,8:24]
                        Q_avg_A_2 = test_A_2[:,0:2]
                        if use_unit_vector:
                                Q_avg_A_2 = norm.get_unit_vector(Q_avg_A_2)
                        psi_truth_A_2 = test_A_2[:,3]
                        if use_neutrons_2:
                                neutrons_A_smeared_2 = process.blur_neutron(test_A_2[:,7])        
                elif (model_2_is_qq):                        
                        model_2 = keras.models.load_model(model_path_2 + f'model{model_type_2}_{model_num_2}_{model_loss_2}.h5',compile = False)
                        test_A_2 = pd.read_pickle(data_path_2 + data_file_2).to_numpy()
                        test_X_A_2 = test_A_2[:,6:22]
                        pt_nuc_A_2 = test_A_2[:,5] * 1000
                        if do_z_norm_2:
                                rpdSignals_A_2 = np.load(data_path_2 + data_znorm)
                        else:
                                rpdSignals_A_2 = test_A_2[:,6:22]
                        Q_avg_A_2 = test_A_2[:,1:3]
                        psi_truth_A_2 = test_A_2[:,5]
                        if use_neutrons_2:
                                neutrons_A_smeared_2 = process.blur_neutron(test_A_2[:,0])
                        
                if model_type_2 == 'cnn' or model_type_2 == 'cnn_test':
                        #reshape from (None, 16) to (None, 6, 6, 1)
                        test_X_A_2 = process.reshape_signal(test_X_A_2)
                        
                if use_neutrons_2:
                        Q_predicted_A_2 = model_2.predict([neutrons_A_smeared_2.astype('float'), test_X_A_2.astype('float')])
                else:
                        Q_predicted_A_2 = model_2.predict([test_X_A_2.astype('float')])
                psi_rec_A_2 = np.arctan2(Q_predicted_A_2[:,1],Q_predicted_A_2[:,0])
                psi_gen_A_2 = np.arctan2(Q_avg_A_2[:,1],Q_avg_A_2[:,0])
                psi_gen_res_model2 = process.GetPsiResidual_np(psi_gen_A_2, psi_rec_A_2)
                psi_truth_res_model2 = process.GetPsiResidual_np(psi_truth_A_2, psi_rec_A_2)
                print("model2: residual: " , psi_gen_res_model2)
                
        Path(output_path).mkdir(parents=True, exist_ok=True)
        # integrated residuals (not broken up between # of neutrons and pt_nuc)
        vis_root.PlotResiduals(psi_gen_rec_A, "psi_gen_rec", output_path)
        vis_root.PlotResiduals(psi_truth_rec_A, "psi_truth_rec", output_path)
        if do_ratio_plot_ptnuc:
#                vis_root.PlotRatio_ptnuc_hist(pt_nuc_A, psi_gen_rec_A, psi_gen_res_model2, upperRange_gen, model_type_1_label , model_type_2_label, output_path, is_gen = True, save_residuals = False)
                vis_root.PlotRatio_ptnuc_hist( pt_nuc_A, pt_nuc_A_2,  psi_gen_rec_A, psi_gen_res_model2, upperRange_gen, model_type_1_label, model_type_2_label, output_path, is_gen = True, save_residuals = True)
                print("gen_res_model2 " , psi_gen_res_model2)
#               vis_root.PlotRatio_ptnuc_hist(pt_nuc_A, pt_nuc_A_2, psi_gen_rec_A, psi_gen_res_model2, upperRange_gen, model_type_1_label , model_type_2_label, output_path, is_gen = True, save_residuals = False)
#                vis_root.PlotRatio_ptnuc(pt_nuc_A, psi_gen_rec_A, psi_truth_res_model2, upperRange_gen, model_type_1 , model_type_2, output_path, save_residuals = False)
        if do_small_batch:
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_rec_A, upperRange_gen, f"psi_gen_rec_{two_trainer_filename}", output_path, save_residuals = False)
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_rec_A, upperRange_truth, f"psi_truth_rec_{two_trainer_filename}", output_path, save_residuals = False)
        else:
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_rec_A, upperRange_gen, "psi_gen_rec", output_path, save_residuals = False)
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_rec_A, upperRange_truth, "psi_truth_rec", output_path, save_residuals = True)

                if do_2rpds:
                        vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_rec_2A, upperRange_gen, "psi_gen_rec_rpd2", output_path, save_residuals = False)
                        vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_rec_2A, upperRange_truth, "psi_truth_rec_rpd2", output_path, save_residuals = True)
                        
                if do_q_res:
                        vis_root.PlotPosResiduals_neutron(neutrons_A, pt_nuc_A, qx_gen_rec_A_unit, upperRange_gen, "qx_unit_gen_rec", output_path, save_residuals = False)
                        vis_root.PlotPosResiduals_neutron(neutrons_A, pt_nuc_A, qy_gen_rec_A_unit, upperRange_gen, "qy_unit_gen_rec", output_path, save_residuals = False)
                        vis_root.PlotPosResiduals_neutron(neutrons_A, pt_nuc_A, qx_gen_rec_A, upperRange_truth, "qx_gen_rec", output_path, save_residuals = False)
                        vis_root.PlotPosResiduals_neutron(neutrons_A, pt_nuc_A, qy_gen_rec_A, upperRange_truth, "qy_gen_rec", output_path, save_residuals = False)

                if do_C_side_res:
                        vis_root.PlotResiduals_neutron(neutrons_B, pt_nuc_B, psi_gen_rec_B, upperRange_gen, "psi_gen_rec_C", output_path, save_residuals = True)
                        vis_root.PlotResiduals_neutron(neutrons_B, pt_nuc_B, psi_truth_rec_B, upperRange_truth, "psi_truth_rec_C", output_path, save_residuals = True)
                if do_AC_res:

#                        vis_root.PlotResiduals_neutron(neutrons_AC, pt_nuc_B, psi_gen_rec_AB, upperRange_gen, "psi_gen_rec_AC", output_path, save_residuals = True)
#                        vis_root.PlotResiduals_neutron(neutrons_AC, pt_nuc_B, psi_truth_rec_AB, upperRange_truth, "psi_truth_rec_AC", output_path, save_residuals = True)
                        vis_root.PlotCosResiduals_neutron(neutrons_AC, pt_nuc_B, psi_rec_diff_AB, 1, 1.01, "AC_det_res",output_path,save_residuals = True)
                        vis_root.PlotCosResiduals_neutron(neutrons_AC, pt_nuc_B, psi_rec_diff_AB, 2, 1.01, "AC_det_res",output_path,save_residuals = False)
                        vis_root.PlotCosResiduals_neutron(neutrons_AC, pt_nuc_B, psi_rec_diff_AB, 3, 1.01, "AC_det_res",output_path,save_residuals = False)


                        if do_2rpds:
                                vis_root.PlotMultiGraphComp_ptnuc(pt_nuc_A, psi_gen_rec_A, psi_gen_rec_2A, psi_gen_rec_AB, psi_gen_rec_2AB, psi_gen_rec_12AB,upperRange_gen,"qp_rpd1", "qp_rpd2", "qp_rpd1_AC", "qp_rpd2_AC", "qp_rpd12_AC", output_path, save_residuals=False)
                                vis_root.PlotResiduals_neutron(neutrons_AC, pt_nuc_B, psi_gen_rec_2AB, upperRange_gen, "psi_gen_rec_AC_rpd2", output_path, save_residuals = True)
                                vis_root.PlotResiduals_neutron(neutrons_AC, pt_nuc_B, psi_truth_rec_2AB, upperRange_truth, "psi_truth_rec_AC_rpd2", output_path, save_residuals = True)
                                print("cos 12_AB")
                                vis_root.PlotCosResiduals_neutron(neutrons_AC, pt_nuc_B, psi_rec_diff_12_AB, 1, 1.01, "AC_det_res_rpd12",output_path,save_residuals = True)
                                print("cos 2_AB")
                                vis_root.PlotCosResiduals_neutron(neutrons_AC, pt_nuc_B, psi_rec_diff_2AB, 1, 1.01, "AC_det_res_rpd2",output_path,save_residuals = True)
 #                               vis_root.PlotCosResiduals_neutron(neutrons_AC, pt_nuc_B, psi_rec_diff_12_AB, 2, 1.01, "AC_det_res_rpd2",output_path,save_residuals = False)
 #                               vis_root.PlotCosResiduals_neutron(neutrons_AC, pt_nuc_B, psi_rec_diff_12_AB, 3, 1.01, "AC_det_res_rpd2",output_path,save_residuals = False)

#                        vis_root.PlotSPResiduals_neutron(neutrons_AC, pt_nuc_B, sp_QVec_rec_AB, 1.01, "AC_SP_det_res",output_path,save_residuals = True)



                        if do_neutron_dep:
                                vis_root.PlotResiduals_neutronDep_1d(neutrons_A, pt_nuc_A, psi_gen_rec_AB, upperRange_gen, "psi_gen_rec", output_path, save_residuals = True)
                        if do_pt_nuc_dep:
                                vis_root.PlotResiduals_ptNucDep_1d(neutrons_A, pt_nuc_A, psi_gen_rec_AB, upperRange_gen, "psi_gen_rec", output_path, save_residuals = True)


        if do_com:
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_com_A, upperRange_gen, "psi_gen_com", output_path, save_residuals = False)
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_com_A, upperRange_truth, "psi_truth_com", output_path,save_residuals = False)
                vis_root.PlotRatio_ptnuc_hist( neutrons_A, pt_nuc_A, psi_gen_rec_A, psi_gen_com_A, 1, upperRange_gen , model_type_1_label, com_label, output_path, is_gen = True, save_residuals = False)

                vis_root.PlotResiduals_neutron(neutrons_B, pt_nuc_B, psi_gen_com_B, upperRange_gen, "psi_gen_com_C", output_path, save_residuals = True)
                vis_root.PlotResiduals_neutron(neutrons_B, pt_nuc_B, psi_truth_com_B, upperRange_truth, "psi_truth_com_C", output_path,save_residuals = True)
                
        if do_model_residuals:
                vis_root.PlotPredictionResiduals(psi_res_model1_model2, model_type, model_type_2, "psi_mod1_mod2", output_path)
                vis_root.PlotPredictionResiduals_neutron(neutrons_A, pt_nuc_A, psi_res_model1_model2, model_type, model_type_2, upperRange_predictionRes, "psi_mod1_mod2", output_path, save_residuals = False)

        if do_ensemble_avg:
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_ensemble_A, upperRange_gen, "psi_gen_ensemble", output_path, save_residuals = False)
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_ensemble_A, upperRange_truth, "psi_truth_ensemble", output_path, save_residuals = False)

        if do_truth_pos_plot:
                vis_root.PlotTruthPos2(test_A[:,1], test_A[:,2], output_path)
        if do_subtracted_channel_plot:
                vis_root.PlotSubtractedChannels(test_A[:,6:22], output_path)

        if do_position_resolution:
                # try again with 400k training; 500k test; 100k valid
                vis_root.PlotCenterTilesPositionRes(Qx_A, Qy_A, psi_gen_rec_A, neutrons_A, pt_nuc_A, "GenPos", output_path, is_gen = True, save_residuals = True)
                vis_root.PlotCenterTilesPositionRes(Q_predicted_A[:,0],Q_predicted_A[:,1], psi_gen_rec_A, neutrons_A, pt_nuc_A, "RecoPos", output_path, is_gen = True, save_residuals = True)
                vis_root.PlotUpperTilesPositionRes(com_A[:,0],com_A[:,1], psi_gen_rec_A, neutrons_A, pt_nuc_A, "ComPos", output_path, is_gen = True, save_residuals = True)
                if do_com:
                        vis_root.PlotPositionRes_reco(Q_avg_A[:,0], Q_avg_A[:,1], psi_gen_com_A, neutrons_A, pt_nuc_A, model_type_1_label, output_path, is_gen = True, save_residuals = True)
        #        vis_root.PlotQPositionRes(Q_avg_A[:,0], Q_avg_A[:,1], q_mag_gen_rec_A, neutrons_A, pt_nuc_A, model_type_1_label, output_path, is_gen = True, save_residuals = True)
#                vis_root.PlotQPos1dRes(Q_avg_A[:,0], Q_avg_A[:,1], qx_gen_rec_A, neutrons_A, pt_nuc_A, "qx_pos_res", output_path, save_residuals = True)
#                vis_root.PlotQPos1dRes(Q_avg_A[:,0], Q_avg_A[:,1], qy_gen_rec_A, neutrons_A, pt_nuc_A, "qy_pos_res", output_path, save_residuals = True)
#                vis_root.PlotQPositionRes_reco(com_A[:,0], com_A[:,1], psi_gen_rec_A, neutrons_A, pt_nuc_A, model_type_1_label, output_path, is_gen = True, save_residuals = True)

                
        print("model1: residual: " , psi_gen_rec_A)
#        print("model2: residual: " , psi_gen_res_model2)
    #
	
