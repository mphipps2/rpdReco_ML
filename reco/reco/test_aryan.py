from pathlib import Path

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
import xgboost as xgb

if __name__ == '__main__':
	debug = False
        model_loss = "rmse"
        model_num = 1
        model_type = "bdt"
        model_type_1_label = "bdt_qqFibers"
        use_neutrons = False
        do_com = False
        do_z_norm = False
        do_small_batch = False
        use_unit_vector = True
        two_trainer_ratio = 0.6
        two_trainer_filename = "40batch"
        do_model_residuals = True
        model_type_2 = "cnn"
        model_type_2_label = "cnn_qqFibers"
        model_num_2 = 1
        model_loss_2 = "mse"
        model_2_is_qRod = True
        model_2_is_qq = False
        do_position_resolution = True
        do_truth_pos_plot = False
        do_subtracted_channel_plot = False
        do_z_norm_2 = False
        use_neutrons_2 = False
        do_ensemble_avg = False
        do_ratio_plot_ptnuc = True
        scenario = "ToyFermi_qqFibers_LHC_noPedNoise/"
        scenario_2 = "ToyFermi_qRods_LHC_noPedNoise/"
        data_path = "../data/"+scenario
        model_path = "../models/"+scenario
        data_path_2 = "../data/"+scenario_2
        model_path_2 = "../models/"+scenario_2
        data_file = "test_A.pickle"
        data_znorm = "test_A_znorm.npy"
        output_path = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Testing/" + scenario + "Model" + str(model_num) + "/"
        upperRange_gen = 1.4
        upperRange_truth = 2.5
        upperRange_predictionRes = 0.5
        # vertical offset in orginal data
        centerX_bias = 0
        centerY_bias = -0.471659

        if do_small_batch:
                
                modelX = xgb.XGBRegressor()
	        modelX.load_model(model_path + f'bdtX{model_type}_{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.h5')
	        modelY = xgb.XGBRegressor()
	        modelY.load_model(model_path + f'bdtY{model_type}_{model_num}_{model_loss}_twotrainer{two_trainer_ratio}.h5')
        else:
                modelX = xgb.XGBRegressor()
	        modelX.load_model(model_path+f'bdtX_{model_num}_{model_loss}.json')
	        modelY = xgb.XGBRegressor()
	        modelY.load_model(model_path+f'bdtY_{model_num}_{model_loss}.json')
        
        test_A = pd.read_pickle(data_path + data_file).to_numpy()

	#set test_x based on model: 6:22 for allchan, 24:32 for avg, 24:26 for CoM
        test_X_A = test_A[:,6:22]
        if model_type == 'cnn' or model_type == 'cnn_test':
                #reshape from (None, 16) to (None, 6, 6, 1)
                test_X_A = process.reshape_signal(test_X_A)
        
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
        recX = modelX.predict(test_X_A.to_numpy(), iteration_range=(0, modelX.best_iteration))
        recY = modelY.predict(test_X_A.to_numpy(), iteration_range=(0, modelY.best_iteration))
        Q_predicted_A = pd.DataFrame([recX,recY],columns = ['Qx','Qy'])

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
                        test_A_2 = pd.read_pickle(data_path_2 + data_file).to_numpy()
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
                        test_A_2 = pd.read_pickle(data_path_2 + data_file).to_numpy()
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
        if do_small_batch:
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_gen_rec_A, upperRange_gen, f"psi_gen_rec_{two_trainer_filename}", output_path, save_residuals = False)
                vis_root.PlotResiduals_neutron(neutrons_A, pt_nuc_A, psi_truth_rec_A, upperRange_truth, f"psi_truth_rec_{two_trainer_filename}", output_path, save_residuals = False)
        else:
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

        if do_truth_pos_plot:
                vis_root.PlotTruthPos(test_A[:,1], test_A[:,2], output_path)
        if do_subtracted_channel_plot:
                vis_root.PlotSubtractedChannels(test_A[:,6:22], output_path)
        if do_ratio_plot_ptnuc:
#                vis_root.PlotRatio_ptnuc_hist(pt_nuc_A, psi_gen_rec_A, psi_gen_res_model2, upperRange_gen, model_type_1_label , model_type_2_label, output_path, is_gen = True, save_residuals = False)
                vis_root.PlotRatio_ptnuc_hist( pt_nuc_A_2, pt_nuc_A, psi_gen_res_model2, psi_gen_rec_A, upperRange_gen, model_type_2_label , model_type_1_label, output_path, is_gen = True, save_residuals = False)
#               vis_root.PlotRatio_ptnuc_hist(pt_nuc_A, pt_nuc_A_2, psi_gen_rec_A, psi_gen_res_model2, upperRange_gen, model_type_1_label , model_type_2_label, output_path, is_gen = True, save_residuals = False)
#                vis_root.PlotRatio_ptnuc(pt_nuc_A, psi_gen_rec_A, psi_truth_res_model2, upperRange_gen, model_type_1 , model_type_2, output_path, save_residuals = False)
        if do_position_resolution:
                # try again with 400k training; 500k test; 100k valid
                vis_root.PlotPositionRes(test_A[:,1], test_A[:,2], psi_gen_rec_A, model_type_1_label, output_path, is_gen = True, save_residuals = True)

        print("model2: residual: " , psi_gen_res_model2)
        print("model1: residual: " , psi_gen_rec_A)