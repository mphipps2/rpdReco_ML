import numpy as np
from root_numpy import array2tree
from ROOT import *

def GaussianFit(h1):
        f1 = TF1('f1','gaus',-np.pi,np.pi)
        f1.SetRange(h1.GetMean()-h1.GetRMS(),h1.GetMean()+h1.GetRMS())
        f1.SetLineColor(2)
        h1.Fit('f1','LR')
        return h1

def GaussianFitGet(h1, idx):
        f1 = TF1('f1','gaus',-np.pi,np.pi)
        f1.SetRange(h1.GetMean()-h1.GetRMS(),h1.GetMean()+h1.GetRMS())
        f1.SetLineColor(2)
        h1.Fit('f1','LR')
        par = f1.GetParameter(idx)
        parError = f1.GetParError(idx)
        return par, parError
        
def MakeTree_1(psi_res, b_name):        
#        return array2tree(psi_res.to_numpy(dtype = [(b_name,np.float64)]))
#        return array2tree(psi_res.to_numpy(dtype = [(b_name,np.float64)]))
        
        #np.array(psi_res, dtype = [(b_name,np.float64)])
        print ("psi_res d type!!! ", psi_res.dtype)

        if "ensemble" in b_name:
                psi_res.dtype = [(b_name,'float32')]
        else:
                psi_res.dtype = [(b_name,'float64')]

        print ("psi_res type " , type(psi_res), " shape: " , psi_res.shape, " dtype " , psi_res.dtype, " dtypeNames " , psi_res.dtype.names, " dtypeFields " , psi_res.dtype.fields)
        print(psi_res)
        return array2tree(psi_res)

def MakeTree_1_predictions(psi_res, b_name):        
#        return array2tree(psi_res.to_numpy(dtype = [(b_name,np.float64)]))
#        return array2tree(psi_res.to_numpy(dtype = [(b_name,np.float64)]))
        
        #np.array(psi_res, dtype = [(b_name,np.float64)])
        print ("psi_res d type!!! ", psi_res.dtype)

        psi_res.dtype = [(b_name,'float32')]
        print ("psi_res type " , type(psi_res), " shape: " , psi_res.shape, " dtype " , psi_res.dtype, " dtypeNames " , psi_res.dtype.names, " dtypeFields " , psi_res.dtype.fields)
        print(psi_res)
        return array2tree(psi_res)


def MakeTree_2(numParticles, pt_nuc, psi_res, b_name):
        print("MakeTree_2 shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
#        numParticles.dtype = [('numParticles','int32')]
        numParticles.dtype = [('numParticles','float64')]
        pt_nuc.dtype = [('pt_nuclear','float64')]
        print ("psi_res d type!!! ", psi_res.dtype)
        psi_res.dtype = [(b_name,'float64')]
#        if "ensemble" in b_name:
#               psi_res.dtype = [(b_name,'float32')]
#        else:
                

        print ("numParticles type " , type(numParticles), " shape: " , numParticles.shape, " dtype " , numParticles.dtype, " dtypeNames " , numParticles.dtype.names, " dtypeFields " , numParticles.dtype.fields)
        print("shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
        myTree = array2tree(numParticles)
        array2tree(pt_nuc, tree = myTree)
        array2tree(psi_res, tree = myTree)
        print('returning tree2')
        return myTree

def MakeTree_2_predictions(numParticles, pt_nuc, psi_res, b_name):
        print("MakeTree_2 shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
#        numParticles.dtype = [('numParticles','int32')]
        numParticles.dtype = [('numParticles','float64')]
        pt_nuc.dtype = [('pt_nuclear','float64')]
        print ("psi_res d type!!! ", psi_res.dtype)
        psi_res.dtype = [(b_name,'float32')]

        print ("numParticles type " , type(numParticles), " shape: " , numParticles.shape, " dtype " , numParticles.dtype, " dtypeNames " , numParticles.dtype.names, " dtypeFields " , numParticles.dtype.fields)
        print("shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
        myTree = array2tree(numParticles)
        array2tree(pt_nuc, tree = myTree)
        array2tree(psi_res, tree = myTree)
        print('returning tree2')
        return myTree
