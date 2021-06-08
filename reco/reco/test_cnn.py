import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
import reco.lib.vis as vis

import tensorflow.keras as keras
from RPD_CM_calculator import RPD_CM_calculator
import matplotlib.pyplot as plt
from Visualization import plot_residual, get_residual_subplot
from Fitting import fit_gaussian, fit_double_gaussian
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({"savefig.bbox": 'tight'})
plt.rcParams['figure.dpi'] = 100
import tensorflow as tf

print('tensorflow version: ', tf.__version__)


def test_cnn():
    uproot_installed = True
    fileName = 'model_21_mseLoss'
    outA = np.load("./Data/test_set/testA.npy", allow_pickle = True)
    outB = np.load("./Data/test_set/testB.npy", allow_pickle = True)

    outA_raw = np.load("./Data/test_set/testA_raw.npy", allow_pickle = True)
    outB_raw = np.load("./Data/test_set/testB_raw.npy", allow_pickle = True)

    """
    ['avgGposX', 'avgGposY', 'avgPx', 'avgPy', 'pt_nuclear', 'RP_true_value',
    'RP_gen_value', 'numParticle', 'channel_0', 'channel_1', 'channel_2',
    'channel_3', 'channel_4', 'channel_5', 'channel_6', 'channel_7',
    'channel_8', 'channel_9', 'channel_10', 'channel_11', 'channel_12',
    'channel_13', 'channel_14', 'channel_15']
    """
    print('outA: ',outA)
    A_signal = outA[:,8:]
    B_signal = outB[:,8:]
    print('A_signal: ',A_signal, " length ",len(A_signal))

    A_inic_q_avg = outA[:, 0:2]
    B_inic_q_avg = outB[:, 0:2]
    print('A_inic_q_avg ', A_inic_q_avg)    
    print('B_inic_q_avg ', B_inic_q_avg)    
        
    A_hit = outA[:, 7]
    B_hit = outB[:, 7]

    A_inic_q_avg_unit_vector = get_unit_vector(A_inic_q_avg)
    B_inic_q_avg_unit_vector = get_unit_vector(B_inic_q_avg)
    
#    A_inic_q_avg_unit_vector = get_unit_vector(A_inic_q_avg, A_hit)
#    B_inic_q_avg_unit_vector = get_unit_vector(B_inic_q_avg, B_hit)

    
    A_ptNuc = outA[:, 4].astype(float)
    B_ptNuc = outB[:, 4].astype(float)
        

#    Apsi_gen = np.arctan2(A_inic_q_avg[:,1],A_inic_q_avg[:,0])
    Apsi_gen = np.arctan2(A_inic_q_avg[:,1],A_inic_q_avg[:,0])
    Apsi_true = outA[:,5].astype(float)


    Bpsi_gen = np.arctan2(B_inic_q_avg[:,1],B_inic_q_avg[:,0])
    Bpsi_true = outB[:,5].astype(float)

    #array gets reformatted but number of events stays same
    A_signal = process_signal(A_signal)
    #    print('processed A_signal: ',A_signal, " length ",len(A_signal))
    B_signal = process_signal(B_signal)

    model = keras.models.load_model(f'./Output/Model/{fileName}.h5', compile = False)
    # model should return 2 values: (QA_x, QA_y)
    QA = model.predict([A_hit.astype(float), A_signal.astype(float)])
    print('QA: ',QA,' length: ',len(QA))
    QB = model.predict([B_hit.astype(float), B_signal.astype(float)])

    dQA  = A_inic_q_avg_unit_vector - QA
    dQB  = B_inic_q_avg_unit_vector - QB

    Adx = dQA[:,0]
    Ady = dQA[:,1]

    Bdx = dQB[:,0]

    Bdy = dQB[:,1]
        
    Apsi_rec = np.arctan2(QA[:,1],QA[:,0])
    Apsi_gen = np.arctan2(A_inic_q_avg[:,1],A_inic_q_avg[:,0])
    AR_rec = np.sqrt(QA[:,0] ** 2 + QA[:, 1] ** 2)

    Bpsi_rec = np.arctan2(QB[:,1],QB[:,0])
    Bpsi_gen = np.arctan2(B_inic_q_avg[:,1],B_inic_q_avg[:,0])
    BR_rec = np.sqrt(QB[:,0] ** 2 + QB[:, 1] ** 2)


    
    #A_inic_q_avg = get_unit_vector(A_inic_q_avg)
    #B_inic_q_avg = get_unit_vector(B_inic_q_avg)
        

    NA, NB, avg_recon_x, avg_recon_y, avg_recon_angle = average_vector(QA, QB)

    NA_gen, NB_gen, avg_gen_x, avg_gen_y, avg_gen_angle = average_vector(A_inic_q_avg, B_inic_q_avg)

    AX = A_inic_q_avg[:,0].astype(float)
    AY = A_inic_q_avg[:, 1].astype(float)
    
    AX_unit = A_inic_q_avg_unit_vector[:,0].astype(float)
    AY_unit = A_inic_q_avg_unit_vector[:, 1].astype(float)
    AR = np.sqrt(AX_unit**2 + AY_unit**2)

    print('AX_unit ', AX_unit )
    
    BX = B_inic_q_avg[:,0].astype(float)
    BY = B_inic_q_avg[:, 1].astype(float)

    BX_unit = B_inic_q_avg_unit_vector[:,0].astype(float)
    BY_unit = B_inic_q_avg_unit_vector[:, 1].astype(float)
    BR = np.sqrt(BX_unit**2 + BY_unit**2)

    ch_A = []
    ch_B = []
    for i in range(16):
        channelNum = i + 8
        ch_A.append(outA[:,channelNum])
        ch_B.append(outB[:,channelNum])
    ch_A_raw = []
    ch_B_raw = []
    for i in range(16):
        channelNum = i + 8
        ch_A_raw.append(outA_raw[:,channelNum])
        ch_B_raw.append(outB_raw[:,channelNum])

    print('A_hit: ',A_hit,' length: ',len(A_hit))
    print("ch_A: ", ch_A)	
    print("ch_A_raw: ", ch_A_raw)	
    TreeA = {"n_incident_neutron": A_hit,
             "X_gen":AX,
             "Y_gen": AY,
             "X_gen_unit":AX_unit,
             "Y_gen_unit": AY_unit,
             "R_gen": AR,
             "Qx_rec": QA[:,0],
             "Qy_rec": QA[:,1],
             "dX": Adx,
             "dY": Ady,
             "psi_true":Apsi_true,
             "psi_rec":Apsi_rec,
             "psi_gen": Apsi_gen,
             "R_rec": AR_rec,
             "Pt_nuc": A_ptNuc,
             "ch_0": ch_A[0],
             "ch_1": ch_A[1],
             "ch_2": ch_A[2],
             "ch_3": ch_A[3],
             "ch_4": ch_A[4],
             "ch_5": ch_A[5],
             "ch_6": ch_A[6],
             "ch_7": ch_A[7],
             "ch_8": ch_A[8],
             "ch_9": ch_A[9],
             "ch_10": ch_A[10],
             "ch_11": ch_A[11],
             "ch_12": ch_A[12],
             "ch_13": ch_A[13],
             "ch_14": ch_A[14],
             "ch_15": ch_A[15],
             "ch_0_raw": ch_A_raw[0],
             "ch_1_raw": ch_A_raw[1],
             "ch_2_raw": ch_A_raw[2],
             "ch_3_raw": ch_A_raw[3],
             "ch_4_raw": ch_A_raw[4],
             "ch_5_raw": ch_A_raw[5],
             "ch_6_raw": ch_A_raw[6],
             "ch_7_raw": ch_A_raw[7],
             "ch_8_raw": ch_A_raw[8],
             "ch_9_raw": ch_A_raw[9],
             "ch_10_raw": ch_A_raw[10],
             "ch_11_raw": ch_A_raw[11],
             "ch_12_raw": ch_A_raw[12],
             "ch_13_raw": ch_A_raw[13],
             "ch_14_raw": ch_A_raw[14],
             "ch_15_raw": ch_A_raw[15]}

    TreeB = {"n_incident_neutron": B_hit,
	     "X_gen":BX,
	     "Y_gen": BY,
             "X_gen_unit":BX_unit,
	     "Y_gen_unit": BY_unit,
	     "R_gen": BR,
	     "Qx_rec": QB[:,0],
	     "Qy_rec": QB[:,1],
	     "dX": Bdx,
	     "dY": Bdy,
	     "psi_true":Bpsi_true,
	     "psi_rec":Bpsi_rec,
	     "psi_gen": Bpsi_gen,
	     "R_rec": BR_rec,
             "Pt_nuc": B_ptNuc,
             "ch_0": ch_B[0],
             "ch_1": ch_B[1],
             "ch_2": ch_B[2],
             "ch_3": ch_B[3],
             "ch_4": ch_B[4],
             "ch_5": ch_B[5],
             "ch_6": ch_B[6],
             "ch_7": ch_B[7],
             "ch_8": ch_B[8],
             "ch_9": ch_B[9],
             "ch_10": ch_B[10],
             "ch_11": ch_B[11],
             "ch_12": ch_B[12],
             "ch_13": ch_B[13],
             "ch_14": ch_B[14],
             "ch_15": ch_B[15],
             "ch_0_raw": ch_B_raw[0],
             "ch_1_raw": ch_B_raw[1],
             "ch_2_raw": ch_B_raw[2],
             "ch_3_raw": ch_B_raw[3],
             "ch_4_raw": ch_B_raw[4],
             "ch_5_raw": ch_B_raw[5],
             "ch_6_raw": ch_B_raw[6],
             "ch_7_raw": ch_B_raw[7],
             "ch_8_raw": ch_B_raw[8],
             "ch_9_raw": ch_B_raw[9],
             "ch_10_raw": ch_B_raw[10],
             "ch_11_raw": ch_B_raw[11],
             "ch_12_raw": ch_B_raw[12],
             "ch_13_raw": ch_B_raw[13],
             "ch_14_raw": ch_B_raw[14],
             "ch_15_raw": ch_B_raw[15]}


    Tree_arms = {"NormAx":NA[0],
		 "NormAy":NA[1],
		 "NormBx": -NB[0],
		 "NormBy": -NB[1],
		 "Average_RP_vector_X": avg_recon_x,
		 "Average_RP_vector_Y": avg_recon_y,
		 "Average_RP_angle": avg_recon_angle}
	
	
    if uproot_installed:
        from ROOT import TFile, TTree
        from array import array
        import uproot
        print('root installed')
        # def fill_array(arr):
        #     output = []
        #     print('arr: ',arr, ' length: ', len(arr))
        #     print('arr.keys(): ',arr.keys())
        #     tree = arr.items()
        #     print('shape of tree: ',np.shape(tree))        
        #     for branch in arr.keys():
        #         print('branch: ',branch)
        #         tmp = array( 'd' )
        #         for i in arr[branch]:
        #             tmp.append(i)
        #         output.append(tmp)
        #     return output
        
        def fill_array(arr):
            output = []
            print('arr: ',arr, ' length: ', len(arr))
            print('arr.keys(): ',arr.keys())
            for branch in arr.keys():
                print('branch: ',branch)
                tmp = array( 'd' )
                for i in arr[branch]:
                    tmp.append(i)
                output.append(tmp)
            return output 
        
        # appears that this is filling the array incorrectly, instead of 200001 events we end up with 12 events from the 12 branches
        print("TreeA type: ",type(TreeA))
        print("TreeA n_incident_neutrons length: ", len(TreeA['n_incident_neutron']))
        nentries = len(TreeA['n_incident_neutron'])
        A = fill_array(TreeA)

        B = fill_array(TreeB)
        arms = fill_array(Tree_arms)

        f = TFile(f'Output/trees/{fileName}.root', 'recreate')

        armA = TTree('ARM A', 'tree')
        armB = TTree('ARM B', 'tree')
        Avg = TTree('Avg RP', 'tree')
        #        print('A: ', A, ' range(len(A)): ', range(len(A)),' len(A) ',len(A))
        print('A type ',type(A))
        print('shape of A: ',np.shape(A))        
        tmpA = [array('d',[0]) for _ in range(len(A))]
        tmpB = [array('d',[0]) for _ in range(len(B))]
        tmparm = [array('d',[0]) for _ in range(len(arms))]
#        print('tmpA[0]: ',tmpA[0])
#        print('tmpA: ',tmpA)
        armA.Branch('n_incident_neutron',tmpA[0], 'n_incident_neutron/D')
#        print('tmpA[1]: ',tmpA[1])
        armA.Branch('X_gen',tmpA[1], 'X_gen/D')
        armA.Branch('Y_gen', tmpA[2], 'Y_gen/D')
        armA.Branch('X_gen_unit',tmpA[3], 'X_gen_unit/D')
        armA.Branch('Y_gen_unit', tmpA[4], 'Y_gen_unit/D')
        armA.Branch('R_gen', tmpA[5], 'R_gen/D')
        armA.Branch('Qx_rec', tmpA[6], 'Qx_rec/D')
        armA.Branch('Qy_rec', tmpA[7], 'Qy_rec/D')
        armA.Branch('dX', tmpA[8], 'dX/D')
        armA.Branch('dY', tmpA[9], 'dY/D')                                
        armA.Branch('psi_true', tmpA[10], 'psi_true/D')        
        armA.Branch('psi_rec', tmpA[11], 'psi_rec/D')
        armA.Branch('psi_gen', tmpA[12], 'psi_gen/D')
        armA.Branch('R_rec', tmpA[13], 'R_rec/D')
        armA.Branch('Pt_nuc', tmpA[14], 'Pt_nuc/D')
        armA.Branch('ch_0', tmpA[15], 'ch_0/D')
        armA.Branch('ch_1', tmpA[16], 'ch_1/D')
        armA.Branch('ch_2', tmpA[17], 'ch_2/D')
        armA.Branch('ch_3', tmpA[18], 'ch_3/D')
        armA.Branch('ch_4', tmpA[19], 'ch_4/D')
        armA.Branch('ch_5', tmpA[20], 'ch_5/D')
        armA.Branch('ch_6', tmpA[21], 'ch_6/D')
        armA.Branch('ch_7', tmpA[22], 'ch_7/D')
        armA.Branch('ch_8', tmpA[23], 'ch_8/D')
        armA.Branch('ch_9', tmpA[24], 'ch_9/D')
        armA.Branch('ch_10', tmpA[25], 'ch_10/D')
        armA.Branch('ch_11', tmpA[26], 'ch_11/D')
        armA.Branch('ch_12', tmpA[27], 'ch_12/D')
        armA.Branch('ch_13', tmpA[28], 'ch_13/D')
        armA.Branch('ch_14', tmpA[29], 'ch_14/D')
        armA.Branch('ch_15', tmpA[30], 'ch_15/D')
        armA.Branch('ch_0_raw', tmpA[31], 'ch_0_raw/D')
        armA.Branch('ch_1_raw', tmpA[32], 'ch_1_raw/D')
        armA.Branch('ch_2_raw', tmpA[33], 'ch_2_raw/D')
        armA.Branch('ch_3_raw', tmpA[34], 'ch_3_raw/D')
        armA.Branch('ch_4_raw', tmpA[35], 'ch_4_raw/D')
        armA.Branch('ch_5_raw', tmpA[36], 'ch_5_raw/D')
        armA.Branch('ch_6_raw', tmpA[37], 'ch_6_raw/D')
        armA.Branch('ch_7_raw', tmpA[38], 'ch_7_raw/D')
        armA.Branch('ch_8_raw', tmpA[39], 'ch_8_raw/D')
        armA.Branch('ch_9_raw', tmpA[40], 'ch_9_raw/D')
        armA.Branch('ch_10_raw', tmpA[41], 'ch_10_raw/D')
        armA.Branch('ch_11_raw', tmpA[42], 'ch_11_raw/D')
        armA.Branch('ch_12_raw', tmpA[43], 'ch_12_raw/D')
        armA.Branch('ch_13_raw', tmpA[44], 'ch_13_raw/D')
        armA.Branch('ch_14_raw', tmpA[45], 'ch_14_raw/D')
        armA.Branch('ch_15_raw', tmpA[46], 'ch_15_raw/D')
        print('tmpB[1]: ',tmpB[1])
        armB.Branch('n_incident_neutron',tmpB[0], 'n_incident_neutron/D')
        armB.Branch('X_gen', tmpB[1], 'X_gen/D')
        armB.Branch('Y_gen', tmpB[2], 'Y_gen/D')
        armB.Branch('X_gen_unit', tmpB[3], 'X_gen_unit/D')
        armB.Branch('Y_gen_unit', tmpB[4], 'Y_gen_unit/D')
        armB.Branch('R_gen', tmpB[5], 'R_gen/D')
        armB.Branch('Qx_rec', tmpB[6], 'Qx_rec/D')
        armB.Branch('Qy_rec', tmpB[7], 'Qy_rec/D')
        armB.Branch('dX', tmpB[8], 'dX/D')
        armB.Branch('dY', tmpB[9], 'dY/D')
        armB.Branch('psi_true', tmpB[10], 'psi_true/D')
        armB.Branch('psi_rec', tmpB[11], 'psi_rec/D')
        armB.Branch('psi_gen',tmpB[12], 'psi_gen/D')
        armB.Branch('R_rec',tmpB[13], 'R_rec/D')
        armB.Branch('Pt_nuc', tmpB[14], 'Pt_nuc/D')
        armB.Branch('ch_0', tmpB[15], 'ch_0/D')
        armB.Branch('ch_1', tmpB[16], 'ch_1/D')
        armB.Branch('ch_2', tmpB[17], 'ch_2/D')
        armB.Branch('ch_3', tmpB[18], 'ch_3/D')
        armB.Branch('ch_4', tmpB[19], 'ch_4/D')
        armB.Branch('ch_5', tmpB[20], 'ch_5/D')
        armB.Branch('ch_6', tmpB[21], 'ch_6/D')
        armB.Branch('ch_7', tmpB[22], 'ch_7/D')
        armB.Branch('ch_8', tmpB[23], 'ch_8/D')
        armB.Branch('ch_9', tmpB[24], 'ch_9/D')
        armB.Branch('ch_10', tmpB[25], 'ch_10/D')
        armB.Branch('ch_11', tmpB[26], 'ch_11/D')
        armB.Branch('ch_12', tmpB[27], 'ch_12/D')
        armB.Branch('ch_13', tmpB[28], 'ch_13/D')
        armB.Branch('ch_14', tmpB[29], 'ch_14/D')
        armB.Branch('ch_15', tmpB[30], 'ch_15/D')
        armB.Branch('ch_0_raw', tmpB[31], 'ch_0_raw/D')
        armB.Branch('ch_1_raw', tmpB[32], 'ch_1_raw/D')
        armB.Branch('ch_2_raw', tmpB[33], 'ch_2_raw/D')
        armB.Branch('ch_3_raw', tmpB[34], 'ch_3_raw/D')
        armB.Branch('ch_4_raw', tmpB[35], 'ch_4_raw/D')
        armB.Branch('ch_5_raw', tmpB[36], 'ch_5_raw/D')
        armB.Branch('ch_6_raw', tmpB[37], 'ch_6_raw/D')
        armB.Branch('ch_7_raw', tmpB[38], 'ch_7_raw/D')
        armB.Branch('ch_8_raw', tmpB[39], 'ch_8_raw/D')
        armB.Branch('ch_9_raw', tmpB[40], 'ch_9_raw/D')
        armB.Branch('ch_10_raw', tmpB[41], 'ch_10_raw/D')
        armB.Branch('ch_11_raw', tmpB[42], 'ch_11_raw/D')
        armB.Branch('ch_12_raw', tmpB[43], 'ch_12_raw/D')
        armB.Branch('ch_13_raw', tmpB[44], 'ch_13_raw/D')
        armB.Branch('ch_14_raw', tmpB[45], 'ch_14_raw/D')
        armB.Branch('ch_15_raw', tmpB[46], 'ch_15_raw/D')
        print('tmparm[1]: ',tmparm[1])
        Avg.Branch('NormAx', tmparm[0], 'NormAx/D')
        Avg.Branch('NormAy', tmparm[1], 'NormAy/D')
        Avg.Branch('NormBx', tmparm[2], 'NormBx/D')
        Avg.Branch('NormBy', tmparm[3], 'NormBy/D')
        Avg.Branch('Average_RP_vector_X', tmparm[4], 'Average_RP_vector_X/D')
        Avg.Branch('Average_RP_vector_Y', tmparm[5], 'Average_RP_vector_Y/D')
        Avg.Branch('Average_RP_angle', tmparm[6], 'Average_RP_angle/D')
#        nentries = len(A.items()['n_incident_neutron'])
#        nentries = len(A )
        print(' nEntries: ', nentries)

        for i in range(nentries):
            if i%10000 == 0:
                print('event: ',i)
            for j in range(47):
                tmpA[j][0] = A[j][i]
            for j in range(47):
                tmpB[j][0] = B[j][i]
            for j in range(7):
                tmparm[j][0] = arms[j][i]
            armA.Fill()
            armB.Fill()
            Avg.Fill()
        f.Write()
        f.Close()
    else:
        np.save("TreeFermi_A.npy", TreeA)
        np.save("TreeFermi_B.npy", TreeB)
        np.save("Tree_arms_fermi.npy", Tree_arms)
