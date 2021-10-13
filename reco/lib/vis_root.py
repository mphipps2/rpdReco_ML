import numpy as np
from ROOT import *
from array import array
# Add package to python path in .bashrc file: eg) export PYTHONPATH='/home/mike/Desktop/rpdReco/reco/'
import lib.ops_root as ops_root
import math

def PlotResiduals(psi_res, b_name, output_dir):

        groupLabels = [22.5, 27.5, 32.5, 37.5]
        gROOT.SetStyle('ATLAS')
        tree = ops_root.MakeTree_1(psi_res, b_name)
        tree.Draw(b_name + '>> h1')
        h1 = gDirectory.Get('h1')
        print( "b_name type: ", type(b_name))
        if "gen" in b_name:
                h1.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
        else:
                h1.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
        h1.GetYaxis().SetTitle('Counts')
        h1.GetYaxis().SetTitleOffset(1.7)
        h1 = ops_root.GaussianFit(h1)

        w = 750
        h = 600
        c2 = TCanvas('c2','c2', w,  h)

        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)

        c2.cd()
        h1.Draw()

#        for i in range(len(groupLabels)-1):
#                l1.DrawLine(25 + i*5, h1.GetYaxis().GetXmin(),25+i*5, h1.GetYaxis().GetXmax())

        c2.cd()
        c2.SetFillColor(kWhite)
        if "gen" in b_name:
                c2.SaveAs(output_dir+f'Residual_GenA.png')
        else:
                c2.SaveAs(output_dir+f'Residual_truthA.png')


def PlotPredictionResiduals(psi_res, model_type, model_type_2, b_name, output_dir):

        groupLabels = [22.5, 27.5, 32.5, 37.5]
        gROOT.SetStyle('ATLAS')

        tree = ops_root.MakeTree_1_predictions(psi_res, b_name)
        tree.Draw(b_name + '>> h1')
        h1 = gDirectory.Get('h1')

        h1.GetXaxis().SetTitle('\Psi_{0}^{CNN}-\Psi_{0}^{FCN}[rad]')
        h1.GetYaxis().SetTitle('Counts')
        h1.GetYaxis().SetTitleOffset(1.7)
        h1 = ops_root.GaussianFit(h1)

        w = 750
        h = 600
        c2 = TCanvas('c2','c2', w,  h)

        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)

        c2.cd()
        h1.Draw()

#        for i in range(len(groupLabels)-1):
#                l1.DrawLine(25 + i*5, h1.GetYaxis().GetXmin(),25+i*5, h1.GetYaxis().GetXmax())

        c2.cd()
        c2.SetFillColor(kWhite)
        c2.SaveAs(output_dir+f'Prediction_Residuals_{model_type}_{model_type_2}.png')

                
def PlotResiduals_neutron(numParticles, pt_nuc, psi_res, upperRange, b_name, output_dir, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        groupLabels = [22.5, 27.5, 32.5, 37.5]
        neutrons = [20, 25, 30, 35, 40]
        pt_nuc_vals = [5, 15, 25, 35, 45]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
        print("plotResiduals_neutron shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
        tree = ops_root.MakeTree_2(numParticles, pt_nuc, psi_res, b_name)
        
        #arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining
        x = array('f',[])
        ex = array('f',[])
        y = array('f',[])
        ey = array('f',[])

        n = len(groupLabels)
        colors = [kBlue, kCyan+1, kSpring, kOrange, kRed]

        for i in range(len(ptLabels)):
                for j in range(len(groupLabels)):
                        nBin = groupLabels[j]
                        x.append(nBin)
                        ex.append(0)
                        lowPt = float(i*10 + 5)
                        highPt = float(i*10 + 15)
                        lowerNeutronCut = TCut(f'numParticles >= {nBin-2}')
                        upperNeutronCut = TCut(f'numParticles < {nBin+3}')
                        lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                        if highPt <= 45:
                                upperPtCut = TCut(f'pt_nuclear < {highPt}')
                        else:
                                upperPtCut = TCut('pt_nuclear < 100000')

                        cut = lowerNeutronCut + upperNeutronCut + lowerPtCut + upperPtCut
                        tree.Draw(b_name + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)

                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()

                        temp_h1 = gDirectory.Get('temp_h1')
                        
                        if "gen" in b_name:
                                temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
                        temp_h1.GetYaxis().SetTitle('Counts')

                        temp_h1.Draw()

                        sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                        y.append(sigma)
                        ey.append(error)
                        print ("sigma ",sigma)
                        text = TPaveText(0.65,0.65,0.9,0.9,"brNDC")
                        if i == len(ptLabels)-1:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons}")
                        else:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons} < "+str(neutrons[i+1]))
                        if j == len(groupLabels)-1:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{nuc}")
                        else:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{nuc} < "+str(pt_nuc_vals[j+1]))

                        text.SetFillStyle(0)
                        text.SetLineColor(0)
                        text.SetShadowColor(0)
                        #                        text.SetBorderSize(1)
 #                       text.SetFillColor(0)
                        text.Draw()
                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if "gen" in b_name:
                                        c2.SaveAs(output_dir+'Residual_{b_name}_A_pt'+str(i)+'neutrons'+str(j)+'.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_{b_name}_A_pt'+str(i)+'neutrons'+str(j)+'.png')


        mg = TMultiGraph()
        
        for i in range(len(ptLabels)):
                color = colors[i]
                tge = TGraphErrors(n, x[i*4:i*4+4], y[i*4:i*4+4], ex[i*4:i*4+4], ey[i*4:i*4+4])
                tge.SetDrawOption('AP')

                lowPt = i*10 + 5
                highPt = i*10 + 15
                tge.SetMarkerColor(color)
                
                if highPt < 55:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}<' + str(highPt))
                else:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}')
                tge.SetLineColor(color)
                tge.SetMarkerStyle(21)
                mg.Add(tge)

        if "gen" in b_name:
                mg.SetTitle(';N_{neutrons};\sigma_{\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}} [rad]')
        else:
                mg.SetTitle(';N_{neutrons};\sigma_{\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}} [rad]')
        mg.GetXaxis().SetLimits(20.,40.)
        mg.GetYaxis().SetRangeUser(0,upperRange)
        mg.GetYaxis().SetLimits(0,upperRange)
        
        w = 750
        h = 600

        c3 = TCanvas('c3','c3', w, h)

        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)

        c3.cd()
        mg.Draw('AP')
        c3.BuildLegend().SetBorderSize(0)
 
        for i in range(len(groupLabels)-1):
                l1.DrawLine(25 + i*5, mg.GetYaxis().GetXmin(),25+i*5, mg.GetYaxis().GetXmax())
        c3.cd()
        c3.SetFillColor(kWhite)
        if "gen" in b_name:
                c3.SaveAs(output_dir+f'NeutronDepResolution_{b_name}_A.png')
        else:
                c3.SaveAs(output_dir+f'NeutronDepResolution_{b_name}_A.png')



def PlotRatio_ptnuc(pt_nuc, psi_res_model1, psi_res_model2, upperRange, b_name1, b_name2, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        pt_nuc_vals = [5, 15, 25, 35, 45]
        group_labels = [10, 20, 30, 40, 50]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
        print("plotResiduals_neutron shape pt_nuc_A " , np.shape(pt_nuc))
        tree1 = ops_root.MakeTree_3(pt_nuc, psi_res_model1, b_name1)
        tree2 = ops_root.MakeTree_3(pt_nuc, psi_res_model2, b_name2)
        
        #arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining

        x = array('f',[])
        ex = array('f',[])
        y_1 = array('f',[])
        ey_1 = array('f',[])
        y_2 = array('f',[])
        ey_2 = array('f',[])

        n = len(group_labels)
        colors = [kBlue, kCyan+1, kSpring, kOrange, kRed]


        for i in range(len(ptLabels)):
                nBin = group_labels[i]
                x.append(nBin)
                ex.append(0)
                lowPt = float(i*10 + 5)
                highPt = float(i*10 + 15)
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut
                tree1.Draw(b_name1 + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)
                
                c2 = TCanvas('c2','c2', 750,  600)
                c2.cd()
                
                temp_h1 = gDirectory.Get('temp_h1')
                
                if is_gen:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
                        
                temp_h1.GetYaxis().SetTitle('Counts')
                temp_h1.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                y_1.append(sigma)
                ey_1.append(error)
                print ("sigma model 1: ",sigma)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c2.SaveAs(output_dir+'Residual_gen_{b_name1}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+'Residual_truth_{b_name1}_A_pt'+str(i)+'.png')

        for i in range(len(ptLabels)):
                lowPt = float(i*10 + 5)
                highPt = float(i*10 + 15)
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut
                tree2.Draw(b_name2 + ' >> temp_h2(50,-TMath::Pi,TMath::Pi)', cut)
                
                c2 = TCanvas('c2','c2', 750,  600)
                c2.cd()
                
                temp_h2 = gDirectory.Get('temp_h2')
                
                if is_gen:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
                        
                temp_h2.GetYaxis().SetTitle('Counts')
                temp_h2.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h2,parameter)
                y_2.append(sigma)
                ey_2.append(error)
                print ("sigma model 1: ",sigma)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c2.SaveAs(output_dir+'Residual_gen_{b_name2}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+'Residual_truth_{b_name2}_A_pt'+str(i)+'.png')


        mg = TMultiGraph()

        color = kRed
        tge_1 = TGraphErrors(n, x[:len(ptLabels)], y_1[:len(ptLabels)], ex[:len(ptLabels)], ey_1[:len(ptLabels)])
        tge_1.SetDrawOption('AP')
        tge_1.SetMarkerColor(color)        
        tge_1.SetTitle(b_name1)
        tge_1.SetLineColor(color)
        tge_1.SetMarkerStyle(kFullCircle)
        mg.Add(tge_1)

        tge_2 = TGraphErrors(n, x[:len(ptLabels)], y_2[:len(ptLabels)], ex[:len(ptLabels)], ey_2[:len(ptLabels)])
        tge_2.SetDrawOption('AP')
        tge_2.SetMarkerColor(color)
        tge_2.SetTitle(b_name2)
        tge_2.SetLineColor(color)
        tge_2.SetMarkerStyle(kOpenCircle)
        mg.Add(tge_2)

        if is_gen:
                mg.SetTitle(';#it{p}_{T}^{nuc};\sigma_{\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}} [rad]')
        else:
                mg.SetTitle(';#it{p}_{T}^{nuc};\sigma_{\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}} [rad]')
        mg.GetXaxis().SetLimits(5.,55)
        mg.GetYaxis().SetRangeUser(0,upperRange)
        mg.GetYaxis().SetLimits(0,upperRange)
        
        #        c3 = TCanvas('c3','c3', 750,  600)
#        c4 = TCanvas('c4','c4', 650,  500)
        c1 = ops_root.MakeRatioCanvas(0.0,0.1,0.2,0.25,0.01)
        
        c1.cd(1)
        print("pad 1; top offset: bottom offset: ", c1.cd(1).GetYlowNDC())
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        mg.Draw('AP')
#        c1.BuildLegend().SetBorderSize(0)
        for i in range(len(group_labels)-1):
                l1.DrawLine(pt_nuc_vals[i+1], mg.GetYaxis().GetXmin(), pt_nuc_vals[i+1], mg.GetYaxis().GetXmax())
        c1.cd(2)
        print("pad 2; bottom offset: ", c1.cd(2).GetYlowNDC())
        tge_ratio = ops_root.myTGraphErrorsDivide(tge_1,tge_2)
#        tge_ratio.SetTitle(';#it{p}_{T}^{nuc}; model 1 / model 2')
#        tge_ratio.SetMarkerColor(kBlack)
#        tge_ratio.SetLineColor(kBlack)
#        tge_ratio.SetMarkerStyle(kFullCircle)
        tge_ratio.Draw('AP')
        if is_gen:
                c1.SaveAs(output_dir+f'ptDepResRatio_gen_{b_name1}_{b_name2}_A_top.png')
        else:
                c1.SaveAs(output_dir+f'ptDepResRatio_truth_Resolution_{b_name1}_{b_name2}_A.png')




def PlotRatio_ptnuc_hist(pt_nuc, pt_nuc_2, psi_res_model1, psi_res_model2, upperRange, b_name1, b_name2, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        pt_nuc_vals = [5, 15, 25, 35, 45]
        pt_nuc_histo_range = [5, 50]
        group_labels = [10, 20, 30, 40, 50]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
        print("plotResiduals_neutron shape pt_nuc_A " , np.shape(pt_nuc))
        tree1 = ops_root.MakeTree_3(pt_nuc, psi_res_model1, b_name1)
        tree2 = ops_root.MakeTree_3(pt_nuc_2, psi_res_model2, b_name2)
        
        #arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining

        x = array('f',[])
        ex = array('f',[])
        y_1 = array('f',[])
        ey_1 = array('f',[])
        y_2 = array('f',[])
        ey_2 = array('f',[])

        n = len(group_labels)
        colors = [kBlue, kCyan+1, kSpring, kOrange, kRed]


        for i in range(len(ptLabels)):
                nBin = group_labels[i]
                x.append(nBin)
                ex.append(0)
                lowPt = float(i*10 + 5)
                highPt = float(i*10 + 15)
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut
                tree1.Draw(b_name1 + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)
                
                c2 = TCanvas('c2','c2', 750,  600)
                c2.cd()
                
                temp_h1 = gDirectory.Get('temp_h1')
                
                if is_gen:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
                        
                temp_h1.GetYaxis().SetTitle('Counts')
                temp_h1.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                y_1.append(sigma)
                ey_1.append(error)
                print ("sigma model 1: ",sigma)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c2.SaveAs(output_dir+'Residual_gen_{b_name1}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+'Residual_truth_{b_name1}_A_pt'+str(i)+'.png')

        for i in range(len(ptLabels)):
                lowPt = float(i*10 + 5)
                highPt = float(i*10 + 15)
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut
                tree2.Draw(b_name2 + ' >> temp_h2(50,-TMath::Pi,TMath::Pi)', cut)
                
                c2 = TCanvas('c2','c2', 750,  600)
                c2.cd()
                
                temp_h2 = gDirectory.Get('temp_h2')
                
                if is_gen:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
                        
                temp_h2.GetYaxis().SetTitle('Counts')
                temp_h2.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h2,parameter)
                y_2.append(sigma)
                ey_2.append(error)
                print ("sigma model 1: ",sigma)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c2.SaveAs(output_dir+'Residual_gen_{b_name2}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+'Residual_truth_{b_name2}_A_pt'+str(i)+'.png')


        c1 = TCanvas("c1", "c1", 550, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
        h1 = TH1D(b_name1, b_name1, len(pt_nuc_vals), pt_nuc_histo_range[0], pt_nuc_histo_range[1])
        h2 = TH1D(b_name2, b_name2, len(pt_nuc_vals), pt_nuc_histo_range[0], pt_nuc_histo_range[1])

        for i in range(len(pt_nuc_vals)):
                h1.SetBinContent(i+1,y_1[i])
                h1.SetBinError(i+1,ey_1[i])
                h2.SetBinContent(i+1,y_2[i])
                h2.SetBinError(i+1,ey_2[i])

        if is_gen:
                h1.SetTitle(';#it{p}_{T}^{nuc};\sigma_{\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}} [rad]')
        else:
                h1.SetTitle(';#it{p}_{T}^{nuc};\sigma_{\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}} [rad]')

        color = kBlue
#        maxratio = 1.04
        maxratio = 1.1
#        minratio = 0.83
        minratio = 0.87
        h1.SetMarkerColor(color)
        h2.SetMarkerColor(color)
        h1.SetLineColor(color)
        h2.SetLineColor(color)
        h1.SetMarkerStyle(kFullCircle)
        h2.SetMarkerStyle(kOpenCircle)
        
        rp = TRatioPlot(h1,h2)
        rp.SetH1DrawOpt("APE")
        rp.SetH2DrawOpt("APE")

        rp.GetLowYaxis().SetNdivisions(505)
        rp.SetLeftMargin(0.15)
        rp.SetLowBottomMargin(0.47)
        rp.SetUpBottomMargin(0.05)

#        rp.SetSeparationMargin(0.005)
        rp.Draw("APE")
#        gPad.Modified()
#        gPad.Update()
 #       rp.SetSplitFraction(0.6)
        rp.GetXaxis().SetTitleOffset(0.95)
        rp.GetLowYaxis().SetTitle("ratio")
        rp.GetUpperRefYaxis().SetTitleOffset(1.45)
        rp.GetUpperRefYaxis().SetRangeUser(0.3,0.95)
        rp.GetLowerRefGraph().SetMinimum(minratio)
        rp.GetLowerRefGraph().SetMaximum(maxratio)

        rp.GetLowerRefGraph().SetLineColor(color)
        rp.GetLowerRefGraph().SetMarkerColor(color)
        
        p = rp.GetUpperPad()
        p.cd()
        l = p.BuildLegend(x1=0.56,y1=0.65,x2=0.88,y2=0.85).SetBorderSize(0)
        p.Modified()
        p.Update()
        c1.Update()        
        if is_gen:
                c1.SaveAs(output_dir+f'ptDepResRatio_gen_{b_name1}_{b_name2}_A_.png')
        else:
                c1.SaveAs(output_dir+f'ptDepResRatio_truth_Resolution_{b_name1}_{b_name2}_A.png')

        







def PlotPositionRes(truth_x, truth_y, psi_res_model1, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        #tile_size = 11.4
        tile_size = 9.6
        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " psi_res shape: ", np.shape(psi_res_model1))
        tree1 = ops_root.MakeTree_4(truth_x, truth_y, psi_res_model1, b_name1)
        
        val = np.zeros((5,5))
        err = np.zeros((5,5))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX}')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY}')
                        cut = x_cut + y_cut
                        is_corner = 0
                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                                is_corner = 1
                        if is_corner:
                                tree1.Draw(b_name1 + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)
                        else:
                                tree1.Draw(b_name1 + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)
                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()
                        print("x_cut: ", x_cut, " y_cut: ", y_cut)
                        temp_h1 = gDirectory.Get('temp_h1')
                
                        if is_gen:
                                temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()

                        sigma, error = ops_root.GaussianFitGet(temp_h1,2)
                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())

                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if is_gen:
                                        c2.SaveAs(output_dir+'Residual_gen_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_truth_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')

        c1 = TCanvas("c1", "c1", 550, 500)
        gStyle.SetPadRightMargin(0.3)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [mm];y [mm]')
        gStyle.SetPalette(kTemperatureMap)
        h1.SetContour(99)
#        gStyle.SetPalette(kBird)
#        gStyle.SetPalette(kViridis)
        h1.Draw("colz")
        c1.Update()
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        l1.DrawLine(pos_center[0], h1.GetYaxis().GetXmin(), pos_center[0], h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[2], h1.GetYaxis().GetXmin(), pos_center[2], h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[4], h1.GetYaxis().GetXmin(), pos_center[4], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[0], h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[2], h1.GetXaxis().GetXmax(), pos_center[2])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[4], h1.GetXaxis().GetXmax(), pos_center[4])

        c1.SaveAs(output_dir+f'positionRes_{b_name1}_.png')        



def PlotTruthPos(truth_x, truth_y, output_dir):
        gROOT.SetStyle('ATLAS')

        tree1 = ops_root.MakeTree_5(truth_x, truth_y)

        gStyle.SetPadRightMargin(0.2)
        tree1.Draw('truth_x:truth_y >> temp_h1(50,-TMath::Pi,TMath::Pi)',"","colz")
        c2 = TCanvas('c2','c2', 750,  600)
        c2.cd()
        temp_h1 = gDirectory.Get('temp_h1')
        
        temp_h1.GetXaxis().SetTitle('x [mm]')
        temp_h1.GetYaxis().SetTitle('y [mm]')
        gStyle.SetPalette(kBird)
        temp_h1.SetContour(99)
        temp_h1.Draw("colz")
        c2.SaveAs(output_dir+'TruthPos.png')
        
def PlotSubtractedChannels(rpd_signal, output_dir):
        gROOT.SetStyle('ATLAS')
        gStyle.SetPadRightMargin(0.2)
        h1 = TH2D("SubtractedChannels_allEvents", "SubtractedChannels_allEvents", 4,-2,2,4,-2,2)
        c2 = TCanvas('c2','c2', 750,  600)
        c2.cd()
        for val in range(np.size(rpd_signal,0)):
                for ch in range(np.size(rpd_signal,1)):
                        x = 0
                        y = 0
                        if ch < 4:
                                y = 1.5
                        elif ch >= 4 and ch < 8:
                                y = 0.5
                        elif ch >= 8 and ch < 12:
                                y = -0.5
                        else:
                                y = -1.5
                        if ch % 4 == 0:
                                x = 1.5
                        elif ch % 4 == 1:
                                x = 0.5
                        elif ch % 4 == 2:
                                x = -0.5
                        elif ch % 4 == 3:
                                x = -1.5
                        h1.Fill(x,y,rpd_signal[val,ch])
        h1.Scale(1./np.size(rpd_signal,0))
        com = np.zeros((1,2))
        total_signal = 0.
        for i in range(4):
                for j in range(4):
                        x = 0
                        y = 0
                        if i == 0: y = -1.5
                        elif (i == 1): y = -0.5
                        elif (i == 2): y = 0.5
                        else: y = 1.5
                        
                        if j == 0: x = -1.5
                        elif j == 1: x = -0.5
                        elif j == 2: x = 0.5
                        elif j == 3: x = 1.5
                        
                        com[:,0] += x*h1.GetBinContent(j+1,i+1)
                        com[:,1] += y*h1.GetBinContent(j+1,i+1)
                        total_signal += h1.GetBinContent(j+1,i+1)
        com[:,0] /= total_signal
        com[:,1] /= total_signal
        print("com0 ", com[:,0], " com1 " , com[:,1], " total signal " , total_signal)

        h1.GetXaxis().SetTitle('Tile Pos x')
        h1.GetYaxis().SetTitle('Tile Pos y')
        gStyle.SetPalette(kBird)
        h1.SetContour(99)
        h1.Draw("colz")

        m = TMarker(com[:,0], com[:,1], 29);
        m.SetMarkerColor(kRed);
        m.SetMarkerSize(4);
        m.Draw();
        tex = TLatex();
        tex.SetNDC();
        tex.SetTextFont(43);
        tex.SetTextSize(21);
        tex.SetLineWidth(2);
        com_x = float('%.2g' % com[:,0])
        com_y = float('%.2g' % com[:,1])
        tex.DrawLatex(0.35,0.56,'CoM: ('+str(com_x)+', '+str(com_y)+')' );

        c2.SaveAs(output_dir+'SubtractedChannels.png')
        
def PlotUnsubtractedChannels(rpd_signal, output_dir):
        gROOT.SetStyle('ATLAS')
        gStyle.SetPadRightMargin(0.2)
        h1 = TH2D("UnsubtractedChannels_allEvents", "UnsubtractedChannels_allEvents", 4,-2,2,4,-2,2)
        c2 = TCanvas('c2','c2', 750,  600)
        c2.cd()
        for val in range(np.size(rpd_signal,0)):
                for ch in range(np.size(rpd_signal,1)):
                        x = 0
                        y = 0
                        if ch < 4:
                                y = 1.5
                        elif ch >= 4 and ch < 8:
                                y = 0.5
                        elif ch >= 8 and ch < 12:
                                y = -0.5
                        else:
                                y = -1.5
                        if ch % 4 == 0:
                                x = 1.5
                        elif ch % 4 == 1:
                                x = 0.5
                        elif ch % 4 == 2:
                                x = -0.5
                        elif ch % 4 == 3:
                                x = -1.5
                        h1.Fill(x,y,rpd_signal[val,ch])
        h1.Scale(1./np.size(rpd_signal,0))
        h1.GetXaxis().SetTitle('Tile Pos x')
        h1.GetYaxis().SetTitle('Tile Pos y')
        gStyle.SetPalette(kBird)
        h1.SetContour(99)
        h1.Draw("colz")
        c2.SaveAs(output_dir+'UnsubtractedChannels.png')
        


        
def PlotPredictionResiduals_neutron(numParticles, pt_nuc, psi_res, model_type, model_type_2, upperRange, b_name, output_dir, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        groupLabels = [22.5, 27.5, 32.5, 37.5]
        neutrons = [20, 25, 30, 35, 40]
        pt_nuc_vals = [5, 15, 25, 35, 45]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
        print("plotResiduals_neutron shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
        tree = ops_root.MakeTree_2_predictions(numParticles, pt_nuc, psi_res, b_name)
        
        #arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining
        x = array('f',[])
        ex = array('f',[])
        y = array('f',[])
        ey = array('f',[])

        n = len(groupLabels)
        colors = [kBlue, kCyan+1, kSpring, kOrange, kRed]

        for i in range(len(ptLabels)):
                for j in range(len(groupLabels)):
                        nBin = groupLabels[j]
                        x.append(nBin)
                        ex.append(0)
                        lowPt = float(i*10 + 5)
                        highPt = float(i*10 + 15)
                        lowerNeutronCut = TCut(f'numParticles >= {nBin-2}')
                        upperNeutronCut = TCut(f'numParticles < {nBin+3}')
                        lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                        if highPt <= 45:
                                upperPtCut = TCut(f'pt_nuclear < {highPt}')
                        else:
                                upperPtCut = TCut('pt_nuclear < 100000')

                        cut = lowerNeutronCut + upperNeutronCut + lowerPtCut + upperPtCut
                        tree.Draw(b_name + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)

                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()

                        temp_h1 = gDirectory.Get('temp_h1')
                        
                        temp_h1.GetXaxis().SetTitle('\Psi_{0}^{CNN}-\Psi_{0}^{FCN}[rad]')

                        temp_h1.GetYaxis().SetTitle('Counts')

                        temp_h1.Draw()

                        sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                        y.append(sigma)
                        ey.append(error)
                        print ("sigma ",sigma)
                        text = TPaveText(0.65,0.65,0.9,0.9,"brNDC")
                        if i == len(ptLabels)-1:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons}")
                        else:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons} < "+str(neutrons[i+1]))
                        if j == len(groupLabels)-1:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{nuc}")
                        else:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{nuc} < "+str(pt_nuc_vals[j+1]))

                        text.SetFillStyle(0)
                        text.SetLineColor(0)
                        text.SetShadowColor(0)
                        #                        text.SetBorderSize(1)
 #                       text.SetFillColor(0)
                        text.Draw()
                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                c2.SaveAs(output_dir+f'Prediction_Residual_{model_type}_{model_type_2}_A_pt'+str(i)+'neutrons'+str(j)+'.png')


        mg = TMultiGraph()
        
        for i in range(len(ptLabels)):
                color = colors[i]
                tge = TGraphErrors(n, x[i*4:i*4+4], y[i*4:i*4+4], ex[i*4:i*4+4], ey[i*4:i*4+4])
                tge.SetDrawOption('AP')

                Lowpt = i*10 + 5
                highPt = i*10 + 15
                tge.SetMarkerColor(color)
                
                if highPt < 55:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}<' + str(highPt))
                else:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}')
                tge.SetLineColor(color)
                tge.SetMarkerStyle(21)
                mg.Add(tge)


        mg.SetTitle(';N_{neutrons};\sigma_{\Psi_{0}^{CNN}-\Psi_{0}^{FCN}} [rad]')

        mg.GetXaxis().SetLimits(20.,40.)
        mg.GetYaxis().SetRangeUser(0,upperRange)
        mg.GetYaxis().SetLimits(0,upperRange)
        
        w = 750
        h = 600

        c3 = TCanvas('c3','c3', w, h)

        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)

        c3.cd()
        mg.Draw('AP')
        c3.BuildLegend(x1=0.2,y1=0.65,x2=0.5,y2=0.9).SetBorderSize(0)
        for i in range(len(groupLabels)-1):
                l1.DrawLine(25 + i*5, mg.GetYaxis().GetXmin(),25+i*5, mg.GetYaxis().GetXmax())
        c3.cd()
        c3.SetFillColor(kWhite)
        c3.SaveAs(output_dir+f'NeutronDepResolution_{model_type}_{model_type_2}_A.png')

                
def PlotTrainingComp(nEpochs, train, val, ylabel, output_file):
        epochs = range(1, nEpochs + 1)
        plt.figure(0)
        plt.plot(epochs, train_mse, color='black', label='Training set')
        plt.plot(epochs, val_mse, 'b', label='Validation set')
        plt.title('')
        plt.ylim([0.5,1.5*np.min(val_mse)])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(output_file)

def PlotTrainingComp(nEpochs, train_loss, val_loss, loss_function, output_file):

        gROOT.SetStyle('ATLAS')
        gStyle.SetPadLeftMargin(0.15)
        gStyle.SetPadRightMargin(0.1)
        w = 700
        h = 600

        c2 = TCanvas('c2','c2', w,  h)
        c2.cd()
        
        h1 = TH1F("validation_loss","validation_loss",nEpochs,1,nEpochs)
        h2 = TH1F("training_loss","training_loss",nEpochs,1,nEpochs)
        val_loss_max = -1
        loss_min = 1000000
        for i in range(nEpochs):
                print ('setting bin content: i ' , i , ' train_loss ' , train_loss[i])
                h1.SetBinContent(i+1,val_loss[i])
                h2.SetBinContent(i+1,train_loss[i])
                if val_loss[i] > val_loss_max:
                        val_loss_max = val_loss[i]
                if val_loss[i] < loss_min:
                        loss_min = val_loss[i]
                elif train_loss[i] < val_loss[i] and train_loss[i] < loss_min:
                        loss_min = train_loss[i]
                
        h1.GetXaxis().SetTitle('Epoch')
        h1.SetTitle('Training loss')
        h2.SetTitle('Validation loss')
        if loss_function == 'mse':
                h1.GetYaxis().SetTitle('Mean Squared Error')
        elif loss_function == 'mae':
                h1.GetYaxis().SetTitle('Mean Absolute Error')
        else:
                print("don't recognize that loss function. Pleas enter either mse or mae or update vis_root::PlotTrainingComp()")

        h1.GetYaxis().SetTitleOffset(1.5)
#        h1.GetYaxis().SetTitleSize(0.04)
        h1.GetYaxis().SetNdivisions(505)
        h1.SetMarkerColor(kBlack)
        h1.SetLineColor(kBlack)
        h2.SetMarkerColor(kBlue)
        h2.SetLineColor(kBlue)
        h1.GetYaxis().SetRangeUser(loss_min - 0.001,val_loss_max + 0.001)

        h1.Draw('PL')
        h2.Draw('PLsame')
        
        leg2 = TLegend(0.53,0.74,0.8,0.9)
        leg2.SetBorderSize(0)
        leg2.SetTextFont(43)
        leg2.SetTextSize(21)
        leg2.SetFillColor(0)
        leg2.AddEntry(h1,"Validation Loss","plfe")
        leg2.AddEntry(h2,"Training Loss","plfe")
        leg2.Draw();
        c2.SetFillColor(kWhite)
        c2.SaveAs(output_file)
        
