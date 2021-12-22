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
                h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
        else:
                h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
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

        h1.GetXaxis().SetTitle('\Psi_{1}^{CNN}-\Psi_{1}^{FCN} [rad]')
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
        print('inside func ')
        gROOT.SetStyle('ATLAS')
#        gStyle.SetOptFit(0)
        print ('bins')
        bins = 100
        groupLabels = [22.5, 27.5, 32.5, 37.5]
        neutrons = [20, 25, 30, 35, 40]
        pt_nuc_vals = [5, 15, 25, 35, 45]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
        print("plotResiduals_neutron shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
        tree = ops_root.MakeTree_2(numParticles, pt_nuc, psi_res, b_name)
        side = "A"
        if "AC" in b_name:
                side = "AC"
        #arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining
        x = array('f',[])
        ex = array('f',[])
        y = array('f',[])
        ey = array('f',[])
        gStyle.SetPadRightMargin(0.1)
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
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-'+side+'}-\Psi_{1}^{Rec-'+side+'} [rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-'+side+'}-\Psi_{1}^{Rec-'+side+'} [rad]')
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
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec}")
                        else:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec} < "+str(pt_nuc_vals[j+1]))

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
                                        c2.SaveAs(output_dir+f'Residual_{b_name}_{side}_pt'+str(i)+'neutrons'+str(j)+'.png')
                                else:
                                        c2.SaveAs(output_dir+f'Residual_{b_name}_{side}_pt'+str(i)+'neutrons'+str(j)+'.png')


        mg = TMultiGraph()
        
        for i in range(len(ptLabels)):
                color = colors[i]
                tge = TGraphErrors(n, x[i*4:i*4+4], y[i*4:i*4+4], ex[i*4:i*4+4], ey[i*4:i*4+4])
                tge.SetDrawOption('AP')

                lowPt = i*10 + 5
                highPt = i*10 + 15
                tge.SetMarkerColor(color)
                
                if highPt < 55:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}<' + str(highPt))
                else:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}')
                tge.SetLineColor(color)
                tge.SetMarkerStyle(21)
                mg.Add(tge)

        if "gen" in b_name:
                mg.SetTitle(';N_{neutrons};\sigma_{\Psi_{1}^{Gen-'+side+'}-\Psi_{1}^{Rec-'+side+'}} [rad]')
        else:
                mg.SetTitle(';N_{neutrons};\sigma_{\Psi_{1}^{Truth-'+side+'}-\Psi_{1}^{Rec-'+side+'}} [rad]')
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








def PlotResiduals_ptNucDep_1d(numParticles, pt_nuc, psi_res, upperRange, b_name, output_dir, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        pt_nuc_vals = [5, 15, 25, 35, 45]
        pt_nuc_histo_range = [5, 50]
        group_labels = [10, 20, 30, 40, 50]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2

        tree1 = ops_root.MakeTree_2(numParticles, pt_nuc, psi_res, b_name)        
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
                lowerNeutronCut = TCut(f'numParticles >= 20')
                upperNeutronCut = TCut(f'numParticles < 40')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut + lowerNeutronCut + upperNeutronCut
                tree1.Draw(b_name + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)
                
                c2 = TCanvas('c2','c2', 750,  600)
                c2.cd()
                
                temp_h1 = gDirectory.Get('temp_h1')
                

                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-AC}-\Psi_{1}^{Rec-AC} [rad]')
                        
                temp_h1.GetYaxis().SetTitle('Counts')
                temp_h1.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                y_1.append(sigma)
                ey_1.append(error)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        c2.SaveAs(output_dir+f'Residual_gen_1dptNuc_{b_name}_AC_pt'+str(i)+'.png')


        c1 = TCanvas("c1", "c1", 550, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
        tge = TGraphErrors(n, x[0:5], y_1[0:5], ex[0:5], ey_1[0:5])
        tge.SetTitle(';#it{p}_{T}^{spec};\sigma_{\Psi_{1}^{Gen-AC}-\Psi_{1}^{Rec-AC}} [rad]')

        tge.GetYaxis().SetRangeUser(0,upperRange)
        tge.GetYaxis().SetLimits(0,upperRange)

        tge.Fit("pol1")
        f = tge.GetFunction("pol1")
        f.SetLineColor(kRed)
        
        tge.Draw("APE")
        txt = TPaveText(0.61,0.75,0.88,0.81,"brNDC")
        txt.AddText("20 #leq N_{Neutrons} < 40")
        txt.SetTextFont(43);
        txt.SetTextSize(21);
        txt.SetFillStyle(0)
        txt.SetLineColor(0)
        txt.SetShadowColor(0)
        txt.Draw()
        
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)

        for i in range(len(pt_nuc_vals)-1):
                l1.DrawLine(15 + i*10, tge.GetYaxis().GetXmin(),15 + i*10, tge.GetYaxis().GetXmax())
                
        c1.SaveAs(output_dir+f'ptDepRes_1d_gen_{b_name}_AC_.png')



                

def PlotResiduals_neutronDep_1d(numParticles, pt_nuc, psi_res, upperRange, b_name, output_dir, save_residuals = False):
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
        gStyle.SetPadRightMargin(0.1)
        n = len(groupLabels)
        colors = [kBlue, kCyan+1, kSpring, kOrange, kRed]


        for j in range(len(groupLabels)):
                nBin = groupLabels[j]
                x.append(nBin)
                ex.append(0)
                lowPt = float(25)
                lowerNeutronCut = TCut(f'numParticles >= {nBin-2}')
                upperNeutronCut = TCut(f'numParticles < {nBin+3}')
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')

                
                        
                cut = lowerNeutronCut + upperNeutronCut + lowerPtCut
                tree.Draw(b_name + ' >> temp_h1(50,-TMath::Pi,TMath::Pi)', cut)
                
                c2 = TCanvas('c2','c2', 750,  600)
                c2.cd()
                
                temp_h1 = gDirectory.Get('temp_h1')
                        
                if "gen" in b_name:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-AC}-\Psi_{1}^{Rec-AC} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-AC}-\Psi_{1}^{Rec-AC} [rad]')
                temp_h1.GetYaxis().SetTitle('Counts')

                temp_h1.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                y.append(sigma)
                ey.append(error)
                print ("sigma ",sigma)
                text = TPaveText(0.65,0.65,0.9,0.9,"brNDC")
                text.AddText(str(neutrons[j])+" #leq N_{neutrons} < "+str(neutrons[j+1]))

                text.AddText("25 #leq #it{p}_{T}^{spec}")

                text.SetFillStyle(0)
                text.SetLineColor(0)
                text.SetShadowColor(0)
                #                        text.SetBorderSize(1)
                #                       text.SetFillColor(0)
                text.Draw()
                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        c2.SaveAs(output_dir+f'Residual_neutronDep1d_{b_name}_AC_pt'+'neutrons'+str(j)+'.png')
 
#        mg = TMultiGraph()
        
        tge = TGraphErrors(n, x[0:4], y[0:4], ex[0:4], ey[0:4])



        if "gen" in b_name:
                tge.SetTitle(';N_{neutrons};\sigma_{\Psi_{1}^{Gen-AC}-\Psi_{1}^{Rec-AC}} [rad]')
        else:
                tge.SetTitle(';N_{neutrons};\sigma_{\Psi_{1}^{Truth-AC}-\Psi_{1}^{Rec-AC}} [rad]')
        tge.GetXaxis().SetLimits(20.,40.)
        tge.GetYaxis().SetRangeUser(0,upperRange)
        tge.GetYaxis().SetLimits(0,upperRange)
        
        w = 750
        h = 600
        
        gStyle.SetOptFit(111)
        c3 = TCanvas('c3','c3', w, h)

        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)

        c3.cd()
        tge.Draw('APE')
        tge.Fit("pol1")
        f = tge.GetFunction("pol1")
        f.SetLineColor(kRed)
        txt = TPaveText(0.18,0.83,0.5,0.89,"brNDC")
        txt.AddText("25 #leq #it{p}_{T}^{spec} [MeV/c]")
        txt.SetTextFont(43);
        txt.SetTextSize(21);
        txt.SetFillStyle(0)
        txt.SetLineColor(0)
        txt.SetShadowColor(0)
        txt.Draw()
        
        for i in range(len(groupLabels)-1):
                l1.DrawLine(25 + i*5, tge.GetYaxis().GetXmin(),25+i*5, tge.GetYaxis().GetXmax())
        c3.cd()
        c3.SetFillColor(kWhite)
        if "gen" in b_name:
                c3.SaveAs(output_dir+f'NeutronDepResolution_1d_{b_name}_AC.png')
        else:
                c3.SaveAs(output_dir+f'NeutronDepResolution_1d_{b_name}_AC.png')






                

def PlotPosResiduals_neutron(numParticles, pt_nuc, pos_res, upperRange, b_name, output_dir, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        groupLabels = [22.5, 27.5, 32.5, 37.5]
        neutrons = [20, 25, 30, 35, 40]
        pt_nuc_vals = [5, 15, 25, 35, 45]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
#        print("plotResiduals_neutron shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
        tree = ops_root.MakeTree_2(numParticles, pt_nuc, pos_res, b_name)
        
        #arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining
        x = array('f',[])
        ex = array('f',[])
        y = array('f',[])
        ey = array('f',[])
        
        gStyle.SetPadRightMargin(0.1)
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
                        tree.Draw(b_name + ' >> temp_h1(60,-3,3)', cut)

                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()

                        temp_h1 = gDirectory.Get('temp_h1')
                        
                        if "qx" in b_name:
                                temp_h1.GetXaxis().SetTitle('Q_{x}^{Gen-A}-Q_{x}^{Rec-A} [mm]')
                        else:
                                temp_h1.GetXaxis().SetTitle('Q_{y}^{Gen-A}-Q_{y}^{Rec-A} [mm]')
                                
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
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec}")
                        else:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec} < "+str(pt_nuc_vals[j+1]))

                        text.SetFillStyle(0)
                        text.SetLineColor(0)
                        text.SetShadowColor(0)
                        #                        text.SetBorderSize(1)
 #                       text.SetFillColor(0)
                        text.Draw()
                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                c2.SaveAs(output_dir+f'Residual_{b_name}_A_pt'+str(i)+'neutrons'+str(j)+'.png')


        mg = TMultiGraph()
        
        for i in range(len(ptLabels)):
                color = colors[i]
                tge = TGraphErrors(n, x[i*4:i*4+4], y[i*4:i*4+4], ex[i*4:i*4+4], ey[i*4:i*4+4])
                tge.SetDrawOption('AP')

                lowPt = i*10 + 5
                highPt = i*10 + 15
                tge.SetMarkerColor(color)
                
                if highPt < 55:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}<' + str(highPt))
                else:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}')
                tge.SetLineColor(color)
                tge.SetMarkerStyle(21)
                mg.Add(tge)

        if "qx" in b_name:
                mg.SetTitle(';N_{neutrons};\sigma_{Q_{x}^{Gen-A}-Q_{x}^{Rec-A}} [mm]')
        else:
                mg.SetTitle(';N_{neutrons};\sigma_{Q_{y}^{Gen-A}-Q_{y}^{Rec-A}} [mm]')
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
        c3.BuildLegend (x1=0.55,y1=0.69,x2=0.88,y2=0.89).SetBorderSize(0)
 
        for i in range(len(groupLabels)-1):
                l1.DrawLine(25 + i*5, mg.GetYaxis().GetXmin(),25+i*5, mg.GetYaxis().GetXmax())
        c3.cd()
        c3.SetFillColor(kWhite)
        c3.SaveAs(output_dir+f'NeutronDepResolution_{b_name}_A.png')








def PlotCosResiduals_neutron(numParticles, pt_nuc, psi_res, k, upperRange, b_name, output_dir, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        groupLabels = [22.5, 27.5, 32.5, 37.5]
        neutrons = [20, 25, 30, 35, 40]
        pt_nuc_vals = [5, 15, 25, 35, 45]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
#        print("plotResiduals_neutron shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))
        psi_res = np.cos(k*psi_res)
        tree = ops_root.MakeTree_2_cos(numParticles, pt_nuc, psi_res, b_name)
        
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
                        tree.Draw(b_name + ' >> temp_h1(50,-1,1)', cut)

                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()

                        temp_h1 = gDirectory.Get('temp_h1')
                        
                        if k == 2:
                                temp_h1.GetXaxis().SetTitle('cos(2(\Psi_{1}^{Rec-A}-\Psi_{1}^{Rec-C}))')
                        elif k == 3:
                                temp_h1.GetXaxis().SetTitle('cos(3(\Psi_{1}^{Rec-A}-\Psi_{1}^{Rec-C}))')
                        elif k == 1:
                                temp_h1.GetXaxis().SetTitle('cos(\Psi_{1}^{Rec-A}-\Psi_{1}^{Rec-C})')

                        temp_h1.GetYaxis().SetTitle('Counts')

                        temp_h1.Draw()

                        cos_mean = temp_h1.GetMean()
                        cos_error = temp_h1.GetMeanError()
                        multConst = sqrt(cos_mean) / cos_mean 
                        y.append(sqrt(cos_mean))
                        ey.append(cos_error*multConst)
                        text = TPaveText(0.6,0.7,0.85,0.9,"brNDC")
                        if i == len(ptLabels)-1:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons}")
                        else:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons} < "+str(neutrons[i+1]))
                        if j == len(groupLabels)-1:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec}")
                        else:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec} < "+str(pt_nuc_vals[j+1]))

                        text.SetFillStyle(0)
                        text.SetLineColor(0)
                        text.SetShadowColor(0)
                        #                        text.SetBorderSize(1)
 #                       text.SetFillColor(0)
                        text.Draw()
                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                c2.SaveAs(output_dir+f'CosResidual_{b_name}_AC_k{k}_pt'+str(i)+'neutrons'+str(j)+'.png')

        mg = TMultiGraph()
        
        for i in range(len(ptLabels)):
                color = colors[i]
                tge = TGraphErrors(n, x[i*4:i*4+4], y[i*4:i*4+4], ex[i*4:i*4+4], ey[i*4:i*4+4])
                tge.SetDrawOption('AP')

                lowPt = i*10 + 5
                highPt = i*10 + 15
                tge.SetMarkerColor(color)
                
                if highPt < 55:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}<' + str(highPt))
                else:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}')
                tge.SetLineColor(color)
                tge.SetMarkerStyle(21)
                mg.Add(tge)
                if k==1:
                        mg.SetTitle(';N_{neutrons};Res(\Psi_{1}^{A|C})')
                elif k==2:
                        mg.SetTitle(';N_{neutrons};Res(2\Psi_{1}^{A|C})')
                elif k==3:
                        mg.SetTitle(';N_{neutrons};Res(3\Psi_{1}^{A|C})')

        mg.GetXaxis().SetLimits(20.,40.)
        mg.GetYaxis().SetRangeUser(0,upperRange)
        mg.GetYaxis().SetLimits(0,upperRange)
        
        w = 750
        h = 600

        print(b_name, " COS RESOLUTIONS k " , k , " " , y)
        print(b_name, " COS errors k " , k , " " , ey)
        c3 = TCanvas('c3','c3', w, h)

        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)

        c3.cd()
        mg.Draw('AP')
        c3.BuildLegend(x1=0.19,y1=0.71,x2=0.52,y2=0.93).SetBorderSize(0)
 
        for i in range(len(groupLabels)-1):
                l1.DrawLine(25 + i*5, mg.GetYaxis().GetXmin(),25+i*5, mg.GetYaxis().GetXmax())
        c3.cd()
        c3.SetFillColor(kWhite)

        c3.SaveAs(output_dir+f'NeutronDepCosResolution_k{k}_{b_name}_A.png')









def PlotSPResiduals_neutron(numParticles, pt_nuc, sp_QVec_rec_AB, upperRange, b_name, output_dir, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        groupLabels = [22.5, 27.5, 32.5, 37.5]
        neutrons = [20, 25, 30, 35, 40]
        pt_nuc_vals = [5, 15, 25, 35, 45]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
#        print("plotResiduals_neutron shape pt_nuc_A " , pt_nuc.shape, " shape numParticlesA " , numParticles, " length ", len(numParticles))


        tree = ops_root.MakeTree_2_cos(numParticles, pt_nuc, sp_QVec_rec_AB, b_name)
        
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
                        tree.Draw(b_name + ' >> temp_h1(200,-2,2)', cut)

                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()

                        temp_h1 = gDirectory.Get('temp_h1')
                        

                        temp_h1.GetXaxis().SetTitle('Q_{2}^{A}Q_{2}^{C}')

                        temp_h1.GetYaxis().SetTitle('Counts')

                        temp_h1.Draw()

                        mean = temp_h1.GetMean()
                        error = temp_h1.GetMeanError()
                        multConst = sqrt(mean) / mean 
                        y.append(sqrt(mean))
                        ey.append(error*multConst)
                        text = TPaveText(0.65,0.65,0.9,0.9,"brNDC")
                        if i == len(ptLabels)-1:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons}")
                        else:
                                text.AddText(str(neutrons[i])+" #leq N_{neutrons} < "+str(neutrons[i+1]))
                        if j == len(groupLabels)-1:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec}")
                        else:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec} < "+str(pt_nuc_vals[j+1]))

                        text.SetFillStyle(0)
                        text.SetLineColor(0)
                        text.SetShadowColor(0)
                        #                        text.SetBorderSize(1)
 #                       text.SetFillColor(0)
                        text.Draw()
                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                c2.SaveAs(output_dir+f'SPResidual_{b_name}_AC_pt'+str(i)+'neutrons'+str(j)+'.png')

        mg = TMultiGraph()
        
        for i in range(len(ptLabels)):
                color = colors[i]
                tge = TGraphErrors(n, x[i*4:i*4+4], y[i*4:i*4+4], ex[i*4:i*4+4], ey[i*4:i*4+4])
                tge.SetDrawOption('AP')

                lowPt = i*10 + 5
                highPt = i*10 + 15
                tge.SetMarkerColor(color)
                
                if highPt < 55:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}<' + str(highPt))
                else:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}')
                tge.SetLineColor(color)
                tge.SetMarkerStyle(21)
                mg.Add(tge)
                mg.SetTitle(';N_{neutrons};\sqrt{< Q_{1}^{Rec-A} \cdot Q_{1}^{Rec-C} >}')

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
        c3.BuildLegend(x1=0.2,y1=0.67,x2=0.49,y2=0.92).SetBorderSize(0)
 
        for i in range(len(groupLabels)-1):
                l1.DrawLine(25 + i*5, mg.GetYaxis().GetXmin(),25+i*5, mg.GetYaxis().GetXmax())
        c3.cd()
        c3.SetFillColor(kWhite)

        c3.SaveAs(output_dir+f'NeutronDepSPResolution_{b_name}_AC.png')











def PlotMultiGraphComp_ptnuc(pt_nuc, psi_res_model1, psi_res_model2, psi_res_model3, psi_res_model4, psi_res_model5, upperRange, b_name1, b_name2, b_name3, b_name4, b_name5, output_dir,  save_residuals = False):
        gROOT.SetStyle('ATLAS')
        is_gen = True
        bins = 100
        pt_nuc_vals = [5, 15, 25, 35, 45]
        group_labels = [10, 20, 30, 40, 50]
        ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
        parameter = 2
        print("plotResiduals_neutron shape pt_nuc_A " , np.shape(pt_nuc))
        tree1 = ops_root.MakeTree_3(pt_nuc, psi_res_model1, b_name1)
        tree2 = ops_root.MakeTree_3(pt_nuc, psi_res_model2, b_name2)
        tree3 = ops_root.MakeTree_3(pt_nuc, psi_res_model3, b_name3)
        tree4 = ops_root.MakeTree_3(pt_nuc, psi_res_model4, b_name4)
        tree5 = ops_root.MakeTree_3(pt_nuc, psi_res_model5, b_name5)
        
        #arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining

        x = array('f',[])
        ex = array('f',[])
        y_1 = array('f',[])
        ey_1 = array('f',[])
        y_2 = array('f',[])
        ey_2 = array('f',[])
        y_3 = array('f',[])
        ey_3 = array('f',[])
        y_4 = array('f',[])
        ey_4 = array('f',[])
        y_5 = array('f',[])
        ey_5 = array('f',[])

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
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                temp_h1.GetYaxis().SetTitle('Counts')
                temp_h1.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                y_1.append(sigma)
                ey_1.append(error)
                print ("sigma error model 1: ",error)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c2.SaveAs(output_dir+f'Residual_gen_{b_name1}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+f'Residual_truth_{b_name1}_A_pt'+str(i)+'.png')

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
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                temp_h2.GetYaxis().SetTitle('Counts')
                temp_h2.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h2,parameter)
                y_2.append(sigma)
                ey_2.append(error)
                print ("sigma error 2: ",error)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c2.SaveAs(output_dir+f'Residual_gen_{b_name2}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+f'Residual_truth_{b_name2}_A_pt'+str(i)+'.png')






        for i in range(len(ptLabels)):
                lowPt = float(i*10 + 5)
                highPt = float(i*10 + 15)
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut
                tree3.Draw(b_name3 + ' >> temp_h3(50,-TMath::Pi,TMath::Pi)', cut)
                
                c3 = TCanvas('c3','c3', 750,  600)
                c3.cd()
                
                temp_h3 = gDirectory.Get('temp_h3')
                
                if is_gen:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                temp_h3.GetYaxis().SetTitle('Counts')
                temp_h3.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h3,parameter)
                y_3.append(sigma)
                ey_3.append(error)
                print ("error model 3: ",error)

                c3.Modified()
                c3.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c3.SaveAs(output_dir+f'Residual_gen_{b_name3}_A_pt'+str(i)+'.png')
                        else:
                                c3.SaveAs(output_dir+f'Residual_truth_{b_name3}_A_pt'+str(i)+'.png')



        for i in range(len(ptLabels)):
                lowPt = float(i*10 + 5)
                highPt = float(i*10 + 15)
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut
                tree4.Draw(b_name4 + ' >> temp_h4(50,-TMath::Pi,TMath::Pi)', cut)
                
                c4 = TCanvas('c4','c4', 750,  600)
                c4.cd()
                
                temp_h4 = gDirectory.Get('temp_h4')
                
                if is_gen:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                temp_h4.GetYaxis().SetTitle('Counts')
                temp_h4.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h4,parameter)
                y_4.append(sigma)
                ey_4.append(error)
                print ("e model 4: ",error)

                c4.Modified()
                c4.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c4.SaveAs(output_dir+f'Residual_gen_{b_name4}_A_pt'+str(i)+'.png')
                        else:
                                c4.SaveAs(output_dir+f'Residual_truth_{b_name4}_A_pt'+str(i)+'.png')





        for i in range(len(ptLabels)):
                lowPt = float(i*10 + 5)
                highPt = float(i*10 + 15)
                lowerPtCut = TCut(f'pt_nuclear >= {lowPt}')
                if highPt <= 45:
                        upperPtCut = TCut(f'pt_nuclear < {highPt}')
                else:
                        upperPtCut = TCut('pt_nuclear < 100000')
                        
                cut = lowerPtCut + upperPtCut
                tree5.Draw(b_name5 + ' >> temp_h5(50,-TMath::Pi,TMath::Pi)', cut)
                
                c5 = TCanvas('c5','c5', 750,  600)
                c5.cd()
                
                temp_h5 = gDirectory.Get('temp_h5')
                
                if is_gen:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                temp_h5.GetYaxis().SetTitle('Counts')
                temp_h5.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h5,parameter)
                y_5.append(sigma)
                ey_5.append(error)
                print ("e model 5: ",error)

                c5.Modified()
                c5.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c5.SaveAs(output_dir+f'Residual_gen_{b_name5}_A_pt'+str(i)+'.png')
                        else:
                                c5.SaveAs(output_dir+f'Residual_truth_{b_name5}_A_pt'+str(i)+'.png')


        mg = TMultiGraph()

        tge_1 = TGraphErrors(n, x[:len(ptLabels)], y_1[:len(ptLabels)], ex[:len(ptLabels)], ey_1[:len(ptLabels)])
        tge_1.SetTitle(b_name1)
        tge_1.SetMarkerStyle(kFullCircle)
        mg.Add(tge_1,'P')
        
        tge_2 = TGraphErrors(n, x[:len(ptLabels)], y_2[:len(ptLabels)], ex[:len(ptLabels)], ey_2[:len(ptLabels)])
        tge_2.SetTitle(b_name2)
        tge_2.SetMarkerStyle(kFullCircle)
        mg.Add(tge_2,'P')
        
        tge_3 = TGraphErrors(n, x[:len(ptLabels)], y_3[:len(ptLabels)], ex[:len(ptLabels)], ey_3[:len(ptLabels)])
        tge_3.SetTitle(b_name3)
        tge_3.SetMarkerStyle(kFullCircle)
        mg.Add(tge_3,'P')
        
        tge_4 = TGraphErrors(n, x[:len(ptLabels)], y_4[:len(ptLabels)], ex[:len(ptLabels)], ey_4[:len(ptLabels)])
        tge_4.SetTitle(b_name4)
        tge_4.SetMarkerStyle(kFullCircle)
        mg.Add(tge_4,'P')

        tge_5 = TGraphErrors(n, x[:len(ptLabels)], y_5[:len(ptLabels)], ex[:len(ptLabels)], ey_5[:len(ptLabels)])
        tge_5.SetTitle(b_name5)
        tge_5.SetMarkerStyle(kFullCircle)
        mg.Add(tge_5,'P')

        
        if is_gen:
                mg.SetTitle(';#it{p}_{T}^{spec} [MeV/c];\sigma_{\Psi_{1}^{Gen}-\Psi_{1}^{Rec}} [rad]')
        else:
                mg.SetTitle(';#it{p}_{T}^{spec} [MeV/c];\sigma_{\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A}} [rad]')
        mg.GetXaxis().SetLimits(5.,55)
        mg.GetYaxis().SetRangeUser(0,upperRange)
        mg.GetYaxis().SetLimits(0,upperRange)
        w = 750
        h = 600

        c6 = TCanvas('c6','c6', w, h)
        
        gStyle.SetPalette(kTemperatureMap)
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        mg.Draw('A PMC')
        c6.BuildLegend(x1=0.61,y1=0.67,x2=0.85,y2=0.92).SetBorderSize(0)
        for i in range(len(group_labels)-1):
                l1.DrawLine(pt_nuc_vals[i+1], mg.GetYaxis().GetXmin(), pt_nuc_vals[i+1], mg.GetYaxis().GetXmax())

        c6.cd()
        c6.SetFillColor(kWhite)

        if is_gen:
                c6.SaveAs(output_dir+f'ptDepResRatio_gen_{b_name1}_{b_name2}_{b_name3}_{b_name4}_{b_name5}.png')
        else:
                c6.SaveAs(output_dir+f'ptDepResRatio_truth_Resolution_{b_name1}_{b_name2}_{b_name3}_{b_name4}_{b_name5}.png')













        
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
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
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
                                c2.SaveAs(output_dir+f'Residual_gen_{b_name1}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+f'Residual_truth_{b_name1}_A_pt'+str(i)+'.png')

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
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
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
                                c2.SaveAs(output_dir+f'Residual_gen_{b_name2}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+f'Residual_truth_{b_name2}_A_pt'+str(i)+'.png')


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
                mg.SetTitle(';#it{p}_{T}^{spec};\sigma_{\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A}} [rad]')
        else:
                mg.SetTitle(';#it{p}_{T}^{spec};\sigma_{\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A}} [rad]')
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
#        tge_ratio.SetTitle(';#it{p}_{T}^{spec}; model 1 / model 2')
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
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                temp_h1.GetYaxis().SetTitle('Counts')
                temp_h1.Draw()

                sigma, error = ops_root.GaussianFitGet(temp_h1,parameter)
                y_1.append(sigma)
                ey_1.append(error)

                c2.Modified()
                c2.SetFillColor(kWhite)
                if save_residuals:
                        if is_gen:
                                c2.SaveAs(output_dir+f'Residual_gen_{b_name1}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+f'Residual_truth_{b_name1}_A_pt'+str(i)+'.png')

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
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                else:
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
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
                                c2.SaveAs(output_dir+f'Residual_gen_{b_name2}_A_pt'+str(i)+'.png')
                        else:
                                c2.SaveAs(output_dir+f'Residual_truth_{b_name2}_A_pt'+str(i)+'.png')


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
                h1.SetTitle(';#it{p}_{T}^{spec};\sigma_{\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A}} [rad]')
        else:
                h1.SetTitle(';#it{p}_{T}^{spec};\sigma_{\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A}} [rad]')

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

        







def PlotPositionRes(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96
        truth_x /= 10
        truth_y /= 10
#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range = [(-9./8.)*tile_size,(-7./8.)*tile_size, (-5./8.)*tile_size, (-3./8.)*tile_size, (-1./8.)*tile_size, (1/8.)*tile_size, (3/8.)*tile_size, (5/8.)*tile_size, (7/8.)*tile_size, (9/8.)*tile_size]
#        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        pos_center = [(-1)*tile_size,(-6./8.)*tile_size,(-4./8.)*tile_size, (-2./8.)*tile_size, (0)*tile_size, (2./8.)*tile_size, (4./8.)*tile_size, (6./8.)*tile_size, (1.)*tile_size]
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " psi_res shape: ", np.shape(psi_res_model1))
        tree1 = ops_root.MakeTree_4(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)
        
#        val = np.zeros((5,5))
        val = np.zeros((9,9))
#        err = np.zeros((5,5))
        err = np.zeros((9,9))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX} && neutrons > 20 && pt_nuc > 30')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
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
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()
                        sigma = temp_h1.GetRMS()
                        error = temp_h1.GetRMSError()
                        #sigma, error = ops_root.GaussianFitGet(temp_h1,2)
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())
#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
                        #if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 6 or y == 7 or y == 8:
                        if math.isnan(sigma):
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error


                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if is_gen:
                                        c2.SaveAs(output_dir+'Residual_gen_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_truth_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                        
        gStyle.SetPadRightMargin(0.15)
        c1 = TCanvas("c1", "c1", 550, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
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
        l1.DrawLine(pos_center[4], h1.GetYaxis().GetXmin(), pos_center[4], h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[8], h1.GetYaxis().GetXmin(), pos_center[8], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[0], h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[4], h1.GetXaxis().GetXmax(), pos_center[4])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[8], h1.GetXaxis().GetXmax(), pos_center[8])

        c1.SaveAs(output_dir+f'positionRes_{b_name1}_.png')        










def PlotFinePositionRes(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96
        # reco has different units from gen
        if "Gen" in b_name1:
                truth_x /= 10
                truth_y /= 10
        
#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range = [(-7./16.)*tile_size, (-5./16.)*tile_size, (-3./16.)*tile_size, (-1./16.)*tile_size, (1/16.)*tile_size, (3/16.)*tile_size, (5/16.)*tile_size, (7/16.)*tile_size]
#        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        pos_center = [(-3./8.)*tile_size,(-2./8.)*tile_size, (-1./8.)*tile_size, (0)*tile_size, (1./8.)*tile_size, (2./8.)*tile_size, (3./8.)*tile_size]
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " psi_res shape: ", np.shape(psi_res_model1))
        if "Reco" in b_name1:
                tree1 = ops_root.MakeTree_4_float32(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)
        else:
                tree1 = ops_root.MakeTree_4(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)

        
#        val = np.zeros((5,5))
        val = np.zeros((7,7))
#        err = np.zeros((5,5))
        err = np.zeros((7,7))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX} && neutrons > 20 && pt_nuc > 30')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
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
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()
                        sigma = temp_h1.GetRMS()
                        error = temp_h1.GetRMSError()
                        #sigma, error = ops_root.GaussianFitGet(temp_h1,2)
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())
#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
                        #if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 6 or y == 7 or y == 8:
                        if math.isnan(sigma):
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error


                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if is_gen:
                                        c2.SaveAs(output_dir+'Residual_gen_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_truth_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                        
        gStyle.SetPadRightMargin(0.2)
        c1 = TCanvas("c1", "c1", 650, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
        h1.GetZaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
        h1.GetZaxis().SetTitleOffset(1.3)
        gStyle.SetPalette(kTemperatureMap)
        h1.SetContour(99)
#        gStyle.SetPalette(kBird)
#        gStyle.SetPalette(kViridis)
        h1.Draw("colz")
        c1.Update()
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        l1.DrawLine(pos_center[3], h1.GetYaxis().GetXmin(), pos_center[3], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[3], h1.GetXaxis().GetXmax(), pos_center[3])

        c1.SaveAs(output_dir+f'positionRes_{b_name1}_.png')        






def PlotCenterTilesPositionRes(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96
        # reco has different units from gen
#        if "Gen" in b_name1:
#                truth_x /= 10
#                truth_y /= 10
        
#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range = [(-15./16.)*tile_size, (-13./16.)*tile_size,(-11./16.)*tile_size, (-9./16.)*tile_size, (-7./16.)*tile_size, (-5./16.)*tile_size, (-3./16.)*tile_size, (-1./16.)*tile_size, (1/16.)*tile_size, (3/16.)*tile_size, (5/16.)*tile_size, (7/16.)*tile_size, (9/16.)*tile_size, (11/16.)*tile_size, (13/16.)*tile_size, (15/16.)*tile_size]
#        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        pos_center = [(-7./8.)*tile_size,(-6./8.)*tile_size,(-5./8.)*tile_size,(-4./8.)*tile_size,(-3./8.)*tile_size,(-2./8.)*tile_size, (-1./8.)*tile_size, (0)*tile_size, (1./8.)*tile_size, (2./8.)*tile_size, (3./8.)*tile_size, (4./8.)*tile_size, (5./8.)*tile_size, (6./8.)*tile_size, (7./8.)*tile_size]
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " psi_res shape: ", np.shape(psi_res_model1))
        if "Reco" in b_name1:
                tree1 = ops_root.MakeTree_4_float32(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)
        else:
                tree1 = ops_root.MakeTree_4(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)

        
#        val = np.zeros((5,5))
        val = np.zeros((15,15))
#        err = np.zeros((5,5))
        err = np.zeros((15,15))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX} && neutrons > 20 && pt_nuc > 30')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
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
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()
                        sigma = temp_h1.GetRMS()
                        error = temp_h1.GetRMSError()
                        #sigma, error = ops_root.GaussianFitGet(temp_h1,2)
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())
#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
                        #if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 6 or y == 7 or y == 8:
                        if math.isnan(sigma):
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error


                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if is_gen:
                                        c2.SaveAs(output_dir+'Residual_gen_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_truth_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                        
        gStyle.SetPadRightMargin(0.2)
        c1 = TCanvas("c1", "c1", 650, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
        h1.GetZaxis().SetTitle('\sigma_{\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]}')
        h1.GetZaxis().SetTitleOffset(1.3)
        gStyle.SetPalette(kTemperatureMap)
        h1.SetContour(99)
#        gStyle.SetPalette(kBird)
#        gStyle.SetPalette(kViridis)
        h1.Draw("colz")
        c1.Update()
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        l1.DrawLine(pos_center[7], h1.GetYaxis().GetXmin(), pos_center[7], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[7], h1.GetXaxis().GetXmax(), pos_center[7])

        c1.SaveAs(output_dir+f'positionRes_{b_name1}_.png')        





def PlotUpperTilesPositionRes(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96
        # reco has different units from gen
        if "Gen" in b_name1:
                truth_x /= 10
                truth_y /= 10
        
#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range_x = [(-15./16.)*tile_size, (-13./16.)*tile_size,(-11./16.)*tile_size, (-9./16.)*tile_size, (-7./16.)*tile_size, (-5./16.)*tile_size, (-3./16.)*tile_size, (-1./16.)*tile_size, (1/16.)*tile_size, (3/16.)*tile_size, (5/16.)*tile_size, (7/16.)*tile_size, (9/16.)*tile_size, (11/16.)*tile_size, (13/16.)*tile_size, (15/16.)*tile_size]
        pos_center_x = [(-7./8.)*tile_size,(-6./8.)*tile_size,(-5./8.)*tile_size,(-4./8.)*tile_size,(-3./8.)*tile_size,(-2./8.)*tile_size, (-1./8.)*tile_size, (0)*tile_size, (1./8.)*tile_size, (2./8.)*tile_size, (3./8.)*tile_size, (4./8.)*tile_size, (5./8.)*tile_size, (6./8.)*tile_size, (7./8.)*tile_size]


        pos_range_y = [(-5./16.)*tile_size,(-3./16.)*tile_size,(-1./16.)*tile_size,(1/16.)*tile_size, (3/16.)*tile_size,(5/16.)*tile_size, (7/16.)*tile_size, (9./16.)*tile_size, (11/16.)*tile_size, (13/16.)*tile_size, (15./16.)*tile_size, (17/16.)*tile_size, (19/16.)*tile_size, (21/16.)*tile_size, (23/16.)*tile_size, (25/16.)*tile_size]
        pos_center_y = [(-2./8.)*tile_size,(-1./8.)*tile_size,(0./8.)*tile_size,(1./8.)*tile_size,(2./8.)*tile_size,(3./8.)*tile_size,(4./8.)*tile_size,(5./8.)*tile_size,(6./8.)*tile_size, (7./8.)*tile_size, (8./8.)*tile_size, (9./8.)*tile_size, (10./8.)*tile_size, (11./8.)*tile_size, (12./8.)*tile_size]
        
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " psi_res shape: ", np.shape(psi_res_model1))
        if "Reco" in b_name1:
                tree1 = ops_root.MakeTree_4_float32(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)
        else:
                tree1 = ops_root.MakeTree_4(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)

        
#        val = np.zeros((5,5))
        val = np.zeros((15,15))
#        err = np.zeros((5,5))
        err = np.zeros((15,15))


        for x in range(len(pos_center_x)):
                for y in range(len(pos_center_x)):
                        lowX = float(pos_range_x[x])                        
                        highX = float(pos_range_x[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX} && neutrons > 20 && pt_nuc > 30')
                        lowY = float(pos_range_y[y])
                        highY = float(pos_range_y[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
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
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()
                        sigma = temp_h1.GetRMS()
                        error = temp_h1.GetRMSError()
                        #sigma, error = ops_root.GaussianFitGet(temp_h1,2)
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())
#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
                        #if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 6 or y == 7 or y == 8:
                        if math.isnan(sigma):
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error


                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if is_gen:
                                        c2.SaveAs(output_dir+'Residual_gen_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_truth_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                        
        gStyle.SetPadRightMargin(0.2)
        c1 = TCanvas("c1", "c1", 650, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center_x), pos_range_x[0], pos_range_x[len(pos_center_x)], len(pos_center_y), pos_range_y[0], pos_range_y[len(pos_center_y)])

        for x in range(len(pos_center_x)):
                for y in range(len(pos_center_x)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
        h1.GetZaxis().SetTitle('\sigma_{\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]}')
        h1.GetZaxis().SetTitleOffset(1.3)
        gStyle.SetPalette(kTemperatureMap)
        h1.SetContour(99)
#        gStyle.SetPalette(kBird)
#        gStyle.SetPalette(kViridis)
        h1.Draw("colz")
        c1.Update()
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        l1.DrawLine(pos_center_x[7], h1.GetYaxis().GetXmin(), pos_center_x[7], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center_y[2], h1.GetXaxis().GetXmax(), pos_center_y[2])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center_y[10], h1.GetXaxis().GetXmax(), pos_center_y[10])

        c1.SaveAs(output_dir+f'positionRes_{b_name1}_.png')        


        

def PlotCoarsePositionRes(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96
        truth_x /= 10
        truth_y /= 10
#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range = [(-9./4.)*tile_size,(-7./4.)*tile_size, (-5./4.)*tile_size, (-3./4.)*tile_size, (-1./4.)*tile_size, (1/4.)*tile_size, (3/4.)*tile_size, (5/4.)*tile_size, (7/4.)*tile_size, (9/4.)*tile_size]
#        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        pos_center = [(-8/4)*tile_size,(-6./4.)*tile_size,(-4./4.)*tile_size, (-2./4.)*tile_size, (0)*tile_size, (2./4.)*tile_size, (4./4.)*tile_size, (6./4.)*tile_size, (8./8.)*tile_size]
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " psi_res shape: ", np.shape(psi_res_model1))
        tree1 = ops_root.MakeTree_4(truth_x, truth_y, psi_res_model1, neutrons, pt_nuc, b_name1)
        
#        val = np.zeros((5,5))
        val = np.zeros((9,9))
#        err = np.zeros((5,5))
        err = np.zeros((9,9))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX} && neutrons > 20 && pt_nuc > 30')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
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
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()
                        sigma = temp_h1.GetRMS()
                        error = temp_h1.GetRMSError()
                        #sigma, error = ops_root.GaussianFitGet(temp_h1,2)
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())
#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
                        #if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 6 or y == 7 or y == 8:
                        if math.isnan(sigma):
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error


                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if is_gen:
                                        c2.SaveAs(output_dir+'Residual_gen_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_truth_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                        
        gStyle.SetPadRightMargin(0.15)
        c1 = TCanvas("c1", "c1", 550, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
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
        l1.DrawLine(pos_center[6], h1.GetYaxis().GetXmin(), pos_center[6], h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[8], h1.GetYaxis().GetXmin(), pos_center[8], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[0], h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[2], h1.GetXaxis().GetXmax(), pos_center[2])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[4], h1.GetXaxis().GetXmax(), pos_center[4])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[6], h1.GetXaxis().GetXmax(), pos_center[6])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[8], h1.GetXaxis().GetXmax(), pos_center[8])

        c1.SaveAs(output_dir+f'positionRes_{b_name1}_.png')        




        

def PlotQPositionRes(truth_x, truth_y, q_mag_model1, neutrons, pt_nuc, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96
 #       truth_x /= 10
 #       truth_y /= 10
#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range = [(-9./8.)*tile_size,(-7./8.)*tile_size, (-5./8.)*tile_size, (-3./8.)*tile_size, (-1./8.)*tile_size, (1/8.)*tile_size, (3/8.)*tile_size, (5/8.)*tile_size, (7/8.)*tile_size, (9/8.)*tile_size]
#        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        pos_center = [(-1)*tile_size,(-6./8.)*tile_size,(-4./8.)*tile_size, (-2./8.)*tile_size, (0)*tile_size, (2./8.)*tile_size, (4./8.)*tile_size, (6./8.)*tile_size, (1.)*tile_size]
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " q_mag shape: ", np.shape(q_mag_model1))
        q_mag_model1_inv = -1 * q_mag_model1
        tree1 = ops_root.MakeTree_4(truth_x, truth_y, q_mag_model1, neutrons, pt_nuc, b_name1)
        
#        val = np.zeros((5,5))
        val = np.zeros((9,9))
#        err = np.zeros((5,5))
        err = np.zeros((9,9))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX} && neutrons > 20 && pt_nuc > 30')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
                                is_corner = 1
                        if is_corner:
                                tree1.Draw(b_name1 + ' >> temp_h1(60,-3,3)', cut)
                        else:
                                tree1.Draw(b_name1 + ' >> temp_h1(60,-3,3)', cut)
                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()
                        print("x_cut: ", x_cut, " y_cut: ", y_cut)
                        temp_h1 = gDirectory.Get('temp_h1')
                

                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()
                        sigma = temp_h1.GetRMS()
                        error = temp_h1.GetRMSError()
#                        sigma, error = ops_root.GaussianFitGet(temp_h1,2)

#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
#                        if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 6 or y == 7 or y == 8:
                        if math.isnan(sigma):
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())

                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                c2.SaveAs(output_dir+'Residual_gen_qMag_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                        
        gStyle.SetPadRightMargin(0.15)
        c1 = TCanvas("c1", "c1", 550, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
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
        l1.DrawLine(pos_center[4], h1.GetYaxis().GetXmin(), pos_center[4], h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[8], h1.GetYaxis().GetXmin(), pos_center[8], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[0], h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[4], h1.GetXaxis().GetXmax(), pos_center[4])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[8], h1.GetXaxis().GetXmax(), pos_center[8])

        c1.SaveAs(output_dir+f'positionRes_qMag_{b_name1}_.png')        





def PlotQPos1dRes(truth_x, truth_y, q_res, neutrons, pt_nuc, b_name1, output_dir, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96
 #       truth_x /= 10
 #       truth_y /= 10
#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range = [(-9./8.)*tile_size,(-7./8.)*tile_size, (-5./8.)*tile_size, (-3./8.)*tile_size, (-1./8.)*tile_size, (1/8.)*tile_size, (3/8.)*tile_size, (5/8.)*tile_size, (7/8.)*tile_size, (9/8.)*tile_size]
#        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        pos_center = [(-1)*tile_size,(-6./8.)*tile_size,(-4./8.)*tile_size, (-2./8.)*tile_size, (0)*tile_size, (2./8.)*tile_size, (4./8.)*tile_size, (6./8.)*tile_size, (1.)*tile_size]
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes truth_x shape: ", np.shape(truth_x), " q_mag shape: ", np.shape(q_res))

        tree1 = ops_root.MakeTree_4(truth_x, truth_y, q_res, neutrons, pt_nuc, b_name1)
        
#        val = np.zeros((5,5))
        val = np.zeros((9,9))
#        err = np.zeros((5,5))
        err = np.zeros((9,9))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'truth_x > {lowX} && truth_x <= {highX} && neutrons > 20 && pt_nuc > 30')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'truth_y > {lowY} && truth_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
                                is_corner = 1
                        if is_corner:
                                tree1.Draw(b_name1 + ' >> temp_h1(60,-3,3)', cut)
                        else:
                                tree1.Draw(b_name1 + ' >> temp_h1(60,-3,3)', cut)
                        c2 = TCanvas('c2','c2', 750,  600)
                        c2.cd()
                        print("x_cut: ", x_cut, " y_cut: ", y_cut)
                        temp_h1 = gDirectory.Get('temp_h1')
                

                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()

                        sigma, error = ops_root.GaussianFitGet(temp_h1,2)
#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:
                        if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 6 or y == 7 or y == 8:
                                sigma = 0
                                error = 0
                        val[x,y] = sigma
                        err[x,y] = error
                        print ("x: ", x , " y: ", y , " sigma: ",sigma, " hRms ", temp_h1.GetRMS())

                        c2.Modified()
                        c2.SetFillColor(kWhite)
                        if save_residuals:
                                if "qx" in b_name1:
                                        c2.SaveAs(output_dir+'Residual_gen_qx_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
                                else:
                                        c2.SaveAs(output_dir+'Residual_gen_qy_posx'+str(x)+'_posy'+str(y)+f'_{b_name1}.png')
        gStyle.SetPadRightMargin(0.15)
        c1 = TCanvas("c1", "c1", 550, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
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
        l1.DrawLine(pos_center[4], h1.GetYaxis().GetXmin(), pos_center[4], h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[8], h1.GetYaxis().GetXmin(), pos_center[8], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[0], h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[4], h1.GetXaxis().GetXmax(), pos_center[4])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[8], h1.GetXaxis().GetXmax(), pos_center[8])
        if "qx" in b_name1:
                c1.SaveAs(output_dir+f'positionRes_qx_{b_name1}_.png')        
        else:
                c1.SaveAs(output_dir+f'positionRes_qy_{b_name1}_.png')        



        


def PlotPositionRes_reco(reco_x, reco_y, psi_res_model1, neutrons, pt_nuc, b_name1, output_dir, is_gen = True, save_residuals = False):
        gROOT.SetStyle('ATLAS')
        bins = 100
        tile_size = 1.14
        #tile_size = .96

#        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_range = [(-9./8.)*tile_size,(-7./8.)*tile_size, (-5./8.)*tile_size, (-3./8.)*tile_size, (-1./8.)*tile_size, (1/8.)*tile_size, (3/8.)*tile_size, (5/8.)*tile_size, (7/8.)*tile_size, (9/8.)*tile_size]
#        pos_center = [(-1)*tile_size, (-0.5)*tile_size, (0)*tile_size, (0.5)*tile_size, (1)*tile_size]
        pos_center = [(-1)*tile_size,(-6./8.)*tile_size,(-4./8.)*tile_size, (-2./8.)*tile_size, (0)*tile_size, (2./8.)*tile_size, (4./8.)*tile_size, (6./8.)*tile_size, (1.)*tile_size]
#        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        pos_center2 = [(-2)*tile_size, (-1)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        print("plotPositionRes reco_x shape: ", np.shape(reco_x), " psi_res shape: ", np.shape(psi_res_model1))
        tree1 = ops_root.MakeTree_7(reco_x, reco_y, psi_res_model1, neutrons, pt_nuc,  b_name1)
        
#        val = np.zeros((5,5))
        val = np.zeros((9,9))
#        err = np.zeros((5,5))
        err = np.zeros((9,9))


        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        lowX = float(pos_range[x])                        
                        highX = float(pos_range[x+1])
                        x_cut = TCut(f'reco_x > {lowX} && reco_x <= {highX} && neutrons > 20 && pt_nuc > 30 ')
                        lowY = float(pos_range[y])
                        highY = float(pos_range[y+1])                                      
                        y_cut = TCut(f'reco_y > {lowY} && reco_y <= {highY} && neutrons > 20 && pt_nuc > 30')
                        cut = x_cut + y_cut
                        is_corner = 0
#                        if (x == 1 and y == 1) or (x == 1 and y == 3) or (x == 3 and y == 1) or (x == 3 and y == 3):
                        if (x == 1 and y == 1) or (x == 1 and y == 5) or (x == 5 and y == 1) or (x == 5 and y == 5):
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
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
                        else:
                                temp_h1.GetXaxis().SetTitle('\Psi_{1}^{Truth-A}-\Psi_{1}^{Rec-A} [rad]')
                        
                        temp_h1.GetYaxis().SetTitle('Counts')
                        temp_h1.Draw()

                        sigma, error = ops_root.GaussianFitGet(temp_h1,2)
                        if math.isnan(sigma) or x == 0 or x == 1 or x == 2 or x == 6 or x == 7 or x == 8 or y == 0 or y == 1 or y == 2 or y == 3 or y == 4 :
                                sigma = 0
                                error = 0
#                        if math.isnan(sigma) or x == 0 or x == 4 or y == 0 or y == 4:


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
                                        
        gStyle.SetPadRightMargin(0.20)
        c1 = TCanvas("c1", "c1", 600, 500)
        gStyle.SetOptStat(0)
        gStyle.SetErrorX(0)
#        h1 = TH2D(b_name1+" model", b_name1+" model", 5, pos_range[0], pos_range[2], 3, pos_range[0], pos_range[2])
        h1 = TH2D(b_name1+" model", b_name1+" model", len(pos_center), pos_range[0], pos_range[len(pos_center)], len(pos_center), pos_range[0], pos_range[len(pos_center)])

        for x in range(len(pos_center)):
                for y in range(len(pos_center)):
                        print("x: ", x, " y: ", y, " val: ", val[x][y], " err: ", err[x][y])
                        h1.SetBinContent(x+1,y+1,val[x][y])
                        h1.SetBinError(x+1,y+1,err[x][y])

        h1.SetTitle(';x [cm];y [cm]')
        h1.GetZaxis().SetTitle('\Psi_{1}^{Gen-A}-\Psi_{1}^{Rec-A} [rad]')
        h1.GetZaxis().SetTitleOffset(1.25)
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
        l1.DrawLine(pos_center[4], h1.GetYaxis().GetXmin(), pos_center[4], h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[8], h1.GetYaxis().GetXmin(), pos_center[8], h1.GetYaxis().GetXmax())
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[0], h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[4], h1.GetXaxis().GetXmax(), pos_center[4])
        l1.DrawLine(h1.GetXaxis().GetXmin(),pos_center[8], h1.GetXaxis().GetXmax(), pos_center[8])

        c1.SaveAs(output_dir+f'positionRes_reco_{b_name1}_.png')        


        

def PlotTruthPos(truth_x, truth_y, tag, output_dir):
        gROOT.SetStyle('ATLAS')
        if "gen" in tag:
                truth_x /= 10
                truth_y /= 10

        tile_size = 1.14
#        tile_size = 0.57
        #tile_size = 0.96
        pos_center = [(-2.)*tile_size,(-1.)*tile_size, (0)*tile_size, (1.)*tile_size,(2.)*tile_size]
        
        tree1 = ops_root.MakeTree_5(truth_x, truth_y)

        gStyle.SetPadRightMargin(0.2)
#        tree1.Draw('truth_x:truth_y >> temp_h1(50,-TMath::Pi,TMath::Pi)',"","colz")
        tree1.Draw('truth_y:truth_x >> temp_h1(200,-2.28,2.28,200,-2.28,2.28)',"","colz")
#        tree1.Draw('truth_y:truth_x >> temp_h1(200,-1.14,1.14,200,-1.14,1.14)',"","colz")
        c2 = TCanvas('c2','c2', 650, 500)
        c2.cd()
        temp_h1 = gDirectory.Get('temp_h1')
        
        temp_h1.GetXaxis().SetTitle('x [cm]')
        temp_h1.GetYaxis().SetTitle('y [cm]')
        temp_h1.GetZaxis().SetTitle('Events')
        temp_h1.GetZaxis().SetTitleOffset(1.3)
        temp_h1.GetXaxis().SetRangeUser(-20,20)
        temp_h1.GetYaxis().SetRangeUser(-20,20)
        gStyle.SetPalette(kBird)
        temp_h1.SetContour(99)
        temp_h1.Draw("colz")
        c2.Update()
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        l1.DrawLine(pos_center[0], temp_h1.GetYaxis().GetXmin(), pos_center[0], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[1], temp_h1.GetYaxis().GetXmin(), pos_center[1], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[2], temp_h1.GetYaxis().GetXmin(), pos_center[2], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[3], temp_h1.GetYaxis().GetXmin(), pos_center[3], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[4], temp_h1.GetYaxis().GetXmin(), pos_center[4], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[0], temp_h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[1], temp_h1.GetXaxis().GetXmax(), pos_center[1])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[2], temp_h1.GetXaxis().GetXmax(), pos_center[2])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[3], temp_h1.GetXaxis().GetXmax(), pos_center[3])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[4], temp_h1.GetXaxis().GetXmax(), pos_center[4])

        c2.SaveAs(output_dir+'TruthPos_'+tag+'.png')

def PlotTruthPos2(truth_x, truth_y, output_dir):
        gROOT.SetStyle('ATLAS')

        tile_size = 1.14
        #tile_size = 0.96
        pos_range = [(-5./4.)*tile_size,(-3./4.)*tile_size, (-1./4.)*tile_size, (1./4.)*tile_size, (3./4.)*tile_size, (5./4.)*tile_size]
        pos_center = [(-2)*tile_size, (-1.)*tile_size, (0)*tile_size, (1.)*tile_size, (2.)*tile_size]
        
        tree1 = ops_root.MakeTree_5(truth_x, truth_y)

        gStyle.SetPadRightMargin(0.2)
#        tree1.Draw('truth_x:truth_y >> temp_h1(50,-TMath::Pi,TMath::Pi)',"","colz")
        tree1.Draw('truth_y:truth_x >> temp_h1(200,-2.28,2.28,200,-2.28,2.28)',"","colz")
        c2 = TCanvas('c2','c2', 750,  600)
        c2.cd()
        temp_h1 = gDirectory.Get('temp_h1')
        
        temp_h1.GetXaxis().SetTitle('x [cm]')
        temp_h1.GetYaxis().SetTitle('y [cm]')
        temp_h1.GetXaxis().SetRangeUser(-20,20)
        temp_h1.GetYaxis().SetRangeUser(-20,20)
        gStyle.SetPalette(kBird)
        temp_h1.SetContour(99)
        temp_h1.Draw("colz")
        c2.Update()
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        l1.DrawLine(pos_center[1], temp_h1.GetYaxis().GetXmin(), pos_center[1], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[2], temp_h1.GetYaxis().GetXmin(), pos_center[2], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[3], temp_h1.GetYaxis().GetXmin(), pos_center[3], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[1], temp_h1.GetXaxis().GetXmax(), pos_center[1])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[2], temp_h1.GetXaxis().GetXmax(), pos_center[2])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[3], temp_h1.GetXaxis().GetXmax(), pos_center[3])

        c2.SaveAs(output_dir+'TruthPos.png')

        
def PlotRecoPos(reco_x, reco_y, tag, output_dir):
        gROOT.SetStyle('ATLAS')
        
        tile_size = 1.14
#        tile_size = 0.57
        #tile_size = 0.96

#        pos_center = [(-1.)*tile_size, (0)*tile_size, (1.)*tile_size]
        pos_center = [(-2.)*tile_size,(-1.)*tile_size, (0)*tile_size, (1.)*tile_size,(2.)*tile_size]

        if "pred" in tag:
                tree1 = ops_root.MakeTree_6_float32(reco_x, reco_y)
        else:
                tree1 = ops_root.MakeTree_6(reco_x, reco_y)
        gStyle.SetPadRightMargin(0.2)
#        tree1.Draw('reco_x:reco_y >> temp_h1(50,-TMath::Pi,TMath::Pi)',"","colz")
        tree1.Draw('reco_y:reco_x >> temp_h1(200,-2.28,2.28,200,-2.28,2.28)',"","colz")
#        tree1.Draw('reco_y:reco_x >> temp_h1(200,-1.14,1.14,200,-1.14,1.14)',"","colz")
        c2 = TCanvas('c2','c2', 650, 500)
        c2.cd()
        temp_h1 = gDirectory.Get('temp_h1')
        
        temp_h1.GetXaxis().SetTitle('x [cm]')
        temp_h1.GetYaxis().SetTitle('y [cm]')
        temp_h1.GetZaxis().SetTitle('Events')
        temp_h1.GetZaxis().SetTitleOffset(1.3)
        gStyle.SetPalette(kBird)
        temp_h1.SetContour(99)
        temp_h1.Draw("colz")
        c2.Update()
        l1 = TLine()
        l1.SetLineStyle(kDashed)
        l1.SetLineColor(kBlack)
        l1.DrawLine(pos_center[0], temp_h1.GetYaxis().GetXmin(), pos_center[0], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[1], temp_h1.GetYaxis().GetXmin(), pos_center[1], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[2], temp_h1.GetYaxis().GetXmin(), pos_center[2], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[3], temp_h1.GetYaxis().GetXmin(), pos_center[3], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(pos_center[4], temp_h1.GetYaxis().GetXmin(), pos_center[4], temp_h1.GetYaxis().GetXmax())
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[0], temp_h1.GetXaxis().GetXmax(), pos_center[0])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[1], temp_h1.GetXaxis().GetXmax(), pos_center[1])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[2], temp_h1.GetXaxis().GetXmax(), pos_center[2])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[3], temp_h1.GetXaxis().GetXmax(), pos_center[3])
        l1.DrawLine(temp_h1.GetXaxis().GetXmin(),pos_center[4], temp_h1.GetXaxis().GetXmax(), pos_center[4])

        c2.SaveAs(output_dir+'RecoPos_'+tag+'.png')

def PlotSubtractedChannels(rpd_signal, output_dir):
        gROOT.SetStyle('ATLAS')
        gStyle.SetPadRightMargin(0.2)
        tile_size = 0.57
#        h1 = TH2D("SubtractedChannels_allEvents", "SubtractedChannels_allEvents", 4,-2,2,4,-2,2)
        h1 = TH2D("SubtractedChannels_allEvents", "SubtractedChannels_allEvents", 4,-1*tile_size*2,tile_size*2,4,-1*tile_size*2,tile_size*2)
        c2 = TCanvas('c2','c2', 750,  600)
        c2.cd()
        for val in range(np.size(rpd_signal,0)):
                for ch in range(np.size(rpd_signal,1)):
                        x = 0
                        y = 0
                        if ch < 4:
                                y = tile_size+(tile_size/2)
                        elif ch >= 4 and ch < 8:
                                y = (tile_size/2)
                        elif ch >= 8 and ch < 12:
                                y = -1 * (tile_size/2)
                        else:
                                y = -1 * (tile_size+(tile_size/2))
                        if ch % 4 == 0:
                                x = tile_size+(tile_size/2)
                        elif ch % 4 == 1:
                                x = (tile_size/2)
                        elif ch % 4 == 2:
                                x = -1 * (tile_size/2)
                        elif ch % 4 == 3:
                                x = -1 * (tile_size+(tile_size/2))
                        h1.Fill(x,y,rpd_signal[val,ch])
        h1.Scale(1./np.size(rpd_signal,0))
        com = np.zeros((1,2))
        total_signal = 0.
        for i in range(4):
                for j in range(4):
                        x = 0
                        y = 0
                        if i == 0: y = -1 * (tile_size+(tile_size/2))
                        elif (i == 1): y = -1 * (tile_size/2)
                        elif (i == 2): y = (tile_size/2)
                        else: y = tile_size+(tile_size/2)
                        
                        if j == 0: x = -1 * (tile_size+(tile_size/2))
                        elif j == 1: x = -1 * (tile_size/2)
                        elif j == 2: x = (tile_size/2)
                        elif j == 3: x = tile_size+(tile_size/2)
                        
                        com[:,0] += x*h1.GetBinContent(j+1,i+1)
                        com[:,1] += y*h1.GetBinContent(j+1,i+1)
                        total_signal += h1.GetBinContent(j+1,i+1)
        com[:,0] /= total_signal
        com[:,1] /= total_signal
        print("com0 ", com[:,0], " com1 " , com[:,1], " total signal " , total_signal)

        h1.GetXaxis().SetTitle('x [cm]')
        h1.GetYaxis().SetTitle('y [cm]')
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
        com_x = float('%.1g' % com[:,0])
        com_y = float('%.3g' % com[:,1])
        tex.DrawLatex(0.35,0.61,'CoM: ('+str(com_x)+', '+str(com_y)+')' );

        c2.SaveAs(output_dir+'SubtractedChannels.png')
        
def PlotUnsubtractedChannels(rpd_signal, output_dir):
        gROOT.SetStyle('ATLAS')
        gStyle.SetPadRightMargin(0.2)
        tile_size = 0.57
        h1 = TH2D("UnsubtractedChannels_allEvents", "UnsubtractedChannels_allEvents", 4,-1*tile_size*2,tile_size*2,4,-1*tile_size*2,tile_size*2)
        c2 = TCanvas('c2','c2', 750,  600)
        c2.cd()
        for val in range(np.size(rpd_signal,0)):
                for ch in range(np.size(rpd_signal,1)):
                        x = 0
                        y = 0
                        if ch < 4:
                                y = tile_size+(tile_size/2)
                        elif ch >= 4 and ch < 8:
                                y = (tile_size/2)
                        elif ch >= 8 and ch < 12:
                                y = -1 * (tile_size/2)
                        else:
                                y = -1 * (tile_size+(tile_size/2))
                        if ch % 4 == 0:
                                x = tile_size+(tile_size/2)
                        elif ch % 4 == 1:
                                x = (tile_size/2)
                        elif ch % 4 == 2:
                                x = -1 * (tile_size/2)
                        elif ch % 4 == 3:
                                x = -1 * (tile_size+(tile_size/2))
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
                        
                        temp_h1.GetXaxis().SetTitle('\Psi_{1}^{CNN}-\Psi_{1}^{FCN} [rad]')

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
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec}")
                        else:
                                text.AddText(str(pt_nuc_vals[j])+" #leq #it{p}_{T}^{spec} < "+str(pt_nuc_vals[j+1]))

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
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}<' + str(highPt))
                else:
                        tge.SetTitle(str(lowPt)+'\leq p_{T}^{spec}')
                tge.SetLineColor(color)
                tge.SetMarkerStyle(21)
                mg.Add(tge)


        mg.SetTitle(';N_{neutrons};\sigma_{\Psi_{1}^{CNN}-\Psi_{1}^{FCN}} [rad]')

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
        elif loss_function == 'rmse':
                h1.GetYaxis().SetTitle('Root Mean Squared Error')
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
        
