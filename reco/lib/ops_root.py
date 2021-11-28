import numpy as np
from root_numpy import array2tree
from ROOT import *
from ctypes import *
import math

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
 #       print ("psi_res d type!!! ", psi_res.dtype)
        print("psi res dtype " , psi_res.dtype)
        print("Pre PSI_RES: " , psi_res)
        psi_res.dtype = [(b_name,'float64')]
        print("post psi res dtype " , psi_res.dtype)
        print("PSI_RES: " , psi_res)
#        if "ensemble" in b_name:
#               psi_res.dtype = [(b_name,'float32')]
#        else:
                

#        print ("numParticles type " , type(numParticles), " shape: " , numParticles.shape, " dtype " , numParticles.dtype, " dtypeNames " , numParticles.dtype.names, " dtypeFields " , numParticles.dtype.fields)
 #       print("shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
 
        myTree = array2tree(numParticles)
        array2tree(pt_nuc, tree = myTree)
        array2tree(psi_res, tree = myTree)
        print('returning tree2')
        return myTree


def MakeTree_2_cos(numParticles, pt_nuc, psi_res, b_name):
        print("MakeTree_2 shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
#        numParticles.dtype = [('numParticles','int32')]
        numParticles.dtype = [('numParticles','float64')]
        pt_nuc.dtype = [('pt_nuclear','float64')]
 #       print ("psi_res d type!!! ", psi_res.dtype)
        print("psi res dtype " , psi_res.dtype)
        print("Pre PSI_RES: " , psi_res)
        psi_res.dtype = [(b_name,'float32')]
        print("post psi res dtype " , psi_res.dtype)
        print("PSI_RES: " , psi_res)
#        if "ensemble" in b_name:
#               psi_res.dtype = [(b_name,'float32')]
#        else:
                

#        print ("numParticles type " , type(numParticles), " shape: " , numParticles.shape, " dtype " , numParticles.dtype, " dtypeNames " , numParticles.dtype.names, " dtypeFields " , numParticles.dtype.fields)
 #       print("shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
 
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

def MakeTree_3(pt_nuc, psi_res, b_name):
        print("MakeTree_2 shape pt_nuc_A " , pt_nuc.shape)
        pt_nuc.dtype = [('pt_nuclear','float64')]
        print ("psi_res d type!!! ", psi_res.dtype)
        psi_res.dtype = [(b_name,'float64')]
#        if "ensemble" in b_name:
#               psi_res.dtype = [(b_name,'float32')]
#        else:
                
        myTree = array2tree(pt_nuc)
        array2tree(psi_res, tree = myTree)
        print('returning tree3')
        return myTree

def MakeTree_4(truth_x, truth_y, psi_res, neutrons, pt_nuc, b_name):
        print('truth_x dtype ' , truth_x.dtype)
        truth_x.dtype = [('truth_x','float64')]
        truth_y.dtype = [('truth_y','float64')]
        print ("psi_res d type!!! ", psi_res.dtype)
        psi_res.dtype = [(b_name,'float64')]
        neutrons.dtype = [('neutrons','float64')]
        pt_nuc.dtype = [('pt_nuc','float64')]
        myTree = array2tree(truth_x)
        array2tree(truth_y, tree = myTree)
        array2tree(psi_res, tree = myTree)
        array2tree(neutrons, tree = myTree)
        array2tree(pt_nuc, tree = myTree)
        print('returning tree4')
        return myTree

def MakeTree_4_float32(truth_x, truth_y, psi_res, neutrons, pt_nuc, b_name):
        print('truth_x dtype ' , truth_x.dtype)
        truth_x.dtype = [('truth_x','float32')]
        truth_y.dtype = [('truth_y','float32')]
        print ("psi_res d type!!! ", psi_res.dtype)
        psi_res.dtype = [(b_name,'float64')]
        neutrons.dtype = [('neutrons','float64')]
        pt_nuc.dtype = [('pt_nuc','float64')]
        myTree = array2tree(truth_x)
        array2tree(truth_y, tree = myTree)
        array2tree(psi_res, tree = myTree)
        array2tree(neutrons, tree = myTree)
        array2tree(pt_nuc, tree = myTree)
        print('returning tree4')
        return myTree


def MakeTree_5(truth_x, truth_y):
        truth_x.dtype = [('truth_x','float64')]
        truth_y.dtype = [('truth_y','float64')]

        myTree = array2tree(truth_x)
        array2tree(truth_y, tree = myTree)
        return myTree

def MakeTree_6(reco_x, reco_y):
        reco_x.dtype = [('reco_x','float64')]
        reco_y.dtype = [('reco_y','float64')]

        myTree = array2tree(reco_x)
        array2tree(reco_y, tree = myTree)
        return myTree

def MakeTree_6_float32(reco_x, reco_y):
        reco_x.dtype = [('reco_x','float32')]
        reco_y.dtype = [('reco_y','float32')]

        myTree = array2tree(reco_x)
        array2tree(reco_y, tree = myTree)
        return myTree


def MakeTree_7(reco_x, reco_y, psi_res, neutrons, pt_nuc, b_name):
        reco_x.dtype = [('reco_x','float64')]
        reco_y.dtype = [('reco_y','float64')]
        print ("psi_res d type!!! ", psi_res.dtype)
        psi_res.dtype = [(b_name,'float64')]
        neutrons.dtype = [('neutrons','float64')]
        pt_nuc.dtype = [('pt_nuc','float64')]
        myTree = array2tree(reco_x)
        array2tree(reco_y, tree = myTree)
        array2tree(psi_res, tree = myTree)
        array2tree(neutrons, tree = myTree)
        array2tree(pt_nuc, tree = myTree)
        print('returning tree4')
        return myTree

def MakeRatioCanvas(leftOffset=0.05, bottomOffset=0.01, leftMargin=0.10, bottomMargin=0.25, edge=0.01):

        c1 = TCanvas('c1','c1', 650,  500)
        rows = 2
        pad = [0] * rows
        Xlow = 0
        Xup = 0
        Ylow = [0] * rows
        Yup = [0] * rows
        PadWidth = (1.0-leftOffset)/((1.0/(1.0-leftMargin)) + (1.0/(1.0-edge)) - 1.0)
        PadHeight = (1.0-bottomOffset)/((1.0/(1.0-bottomMargin)) + (1.0/(1.0-edge)))
        Xlow = leftOffset
        Xup = 1
        Ylow[1] = bottomOffset
        Ylow[0] = 1.0-PadHeight/(1.0-edge) - 0.2
        Yup[1] = bottomOffset + PadHeight/(1.0-bottomMargin) - 0.2
        Yup[0] = 1
        padName = ""

        pad[0] = TPad("p_0", "p_0", Xlow, Ylow[0], Xup, Yup[0])
        pad[1] = TPad("p_1", "p_1", Xlow, Ylow[1], Xup, Yup[1])
  
        c1.cd()
        pad[0].SetLeftMargin(leftMargin)
        pad[0].SetRightMargin(edge)
        pad[0].SetTopMargin(edge)
        pad[0].SetBottomMargin(0)
        pad[0].Draw()
        pad[0].cd()
        pad[0].SetNumber(0)
        print("ops_root pad ", " ; top offset: bottom offset: ", c1.cd(1).GetYlowNDC())
        c1.cd()
        pad[1].SetLeftMargin(leftMargin)
        pad[1].SetRightMargin(edge)
        pad[1].SetTopMargin(0)
        print("ops_root pad here ", 0, " ; top offset: bottom offset: ", c1.cd(1).GetYlowNDC())
        pad[1].SetBottomMargin(bottomMargin)                        
        pad[1].Draw()
        print("ops_root pad heree", 0, " ; top offset: bottom offset: ", c1.cd(1).GetYlowNDC())
#        pad[1].cd()
        print("ops_root pad hereee", 0, " ; top offset: bottom offset: ", c1.cd(1).GetYlowNDC())
        print("bottom ops_root pad 2; bottom offset: ", c1.cd(2).GetYlowNDC())
        pad[1].SetNumber(1)
        print("ops_root pad ", 0, " ; top offset: bottom offset: ", c1.cd(1).GetYlowNDC())
        print("ops_root pad ", 1, " ; top offset: bottom offset: ", c1.cd(1+1).GetYlowNDC())

#        c1.cd(1).SetLeftMargin(leftMargin)
#        c1.cd(2).SetLeftMargin(leftMargin)
        print("bottom ops_root pad 1; top offset: bottom offset: ", c1.cd(1).GetYlowNDC())
        print("bottom ops_root pad 2; bottom offset: ", c1.cd(2).GetYlowNDC())
        return c1

def myTGraphErrorsDivide(g1, g2):

        n1=g1.GetN()
        n2=g2.GetN()
          
        if n1!=n2:
                print("**myTGraphErrorsDivide: objects dont have same number of entries !")                  

        g3 = TGraphErrors()                          
        x1=c_double()
        y1=c_double()
        x2=c_double()
        y2=c_double()
        dx1=0.
        dy1=0.
        dy2=0.          
        iv=0
        
        for i1 in range(n1):
                for i2 in range(n2):
                        g1.GetPoint(i1,x1,y1)
                        g2.GetPoint(i2,x2,y2)
                        print("i1: ", i1, " x1 " , x1, " y1 ", y1)
                        print("i2: ", i2, " x2 " , x2, " y2 ", y2)
                        if math.isclose(x1.value,x2.value):
                                dx1  = g1.GetErrorX(i1)
                                if y1!=0:
                                        dy1  = g1.GetErrorY(i1)/y1.value
                                if y2!=0:
                                        dy2  = g2.GetErrorY(i2)/y2.value
                                if y2!=0.:
                                        g3.SetPoint(iv, x1.value,y1.value/y2.value)
                                else:
                                        g3.SetPoint(iv, x1.value,y2.value)
                                e=0.
                        
                                if y1.value!=0 and y2.value!=0 :
                                        e=math.sqrt(dy1*dy1+dy2*dy2)*(y1.value/y2.value)
                                g3.SetPointError(iv,dx1,e)

                                iv+=1

        return g3

                                  
def setAtlasStyle():
        atlasStyle = TStyle("atlasStyle","atlasStyle")
        icol = 0  # white
        atlasStyle.SetFrameBorderMode(icol)
        atlasStyle.SetFrameFillColor(icol)
        atlasStyle.SetCanvasBorderMode(icol)
        atlasStyle.SetCanvasColor(icol)
        atlasStyle.SetPadBorderMode(icol)
        atlasStyle.SetPadColor(icol)
        atlasStyle.SetStatColor(icol)
        # set the paper & margin sizes
        atlasStyle.SetPaperSize(20,26)
        
        # set margin sizes
        atlasStyle.SetPadTopMargin(0.05)
        atlasStyle.SetPadRightMargin(0.15)
        atlasStyle.SetPadBottomMargin(0.25)
        atlasStyle.SetPadLeftMargin(0.25)
        
        # set title offsets (for axis label)
        atlasStyle.SetTitleXOffset(1.4)
        atlasStyle.SetTitleYOffset(1.4)
        
        # use large fonts
        font=42 # Helvetica
        tsize=0.05
        atlasStyle.SetTextFont(font)
        
        atlasStyle.SetTextSize(tsize)
        atlasStyle.SetLabelFont(font,"x")
        atlasStyle.SetTitleFont(font,"x")
        atlasStyle.SetLabelFont(font,"y")
        atlasStyle.SetTitleFont(font,"y")
        atlasStyle.SetLabelFont(font,"z")
        atlasStyle.SetTitleFont(font,"z")
        
        atlasStyle.SetLabelSize(tsize,"x")
        atlasStyle.SetTitleSize(tsize,"x")
        atlasStyle.SetLabelSize(tsize,"y")
        atlasStyle.SetTitleSize(tsize,"y")
        atlasStyle.SetLabelSize(tsize,"z")
        atlasStyle.SetTitleSize(tsize,"z")
        
        #use bold lines and markers
        atlasStyle.SetMarkerStyle(20)
        atlasStyle.SetMarkerSize(1.2)
        
        # get rid of X error bars
        #atlasStyle.SetErrorX(0.001)
        # get rid of error bar caps
        atlasStyle.SetEndErrorSize(0.)
        
        # do not display any of the standard histogram decorations
        atlasStyle.SetOptTitle(0)
        #atlasStyle.SetOptStat(1111)
        atlasStyle.SetOptStat(0)
        #atlasStyle.SetOptFit(1111)
        atlasStyle.SetOptFit(0)
        
        # put tick marks on top and RHS of plots
        atlasStyle.SetPadTickX(1)
        atlasStyle.SetPadTickY(1)
        gROOT.SetStyle("atlasStyle")
