import numpy as np
from ROOT import *
from array import array
# Add package to python path in .bashrc file: eg) export PYTHONPATH='/home/mike/Desktop/rpdReco/reco/'
import lib.ops_root as ops_root

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
