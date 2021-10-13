import sys

from xgboost.sklearn import XGBClassifier

sys.path.append('/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/rpdreco/reco')

#import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from array import array

from ROOT import *
from root_numpy import array2tree

def normalize(dataset):
	for i in range(len(dataset.columns)):
		dataset.iloc[:,i] = dataset.iloc[:,i]-dataset.iloc[:,i].mean()
		dataset.iloc[:,i] = dataset.iloc[:,i]/dataset.iloc[:,i].std()

	return dataset

def GaussianFit(h1):
	f1 = TF1('f1','gaus',-np.pi,np.pi)
	f1.SetRange(h1.GetMean()-h1.GetRMS(),h1.GetMean()+h1.GetRMS())
	h1.Fit('f1','LR')
	return h1

def GaussianFitGet(h1, idx):
	f1 = TF1('f1','gaus',-np.pi,np.pi)
	f1.SetRange(h1.GetMean()-h1.GetRMS(),h1.GetMean()+h1.GetRMS())
	h1.Fit('f1','LR')
	par = f1.GetParameter(idx)
	parError = f1.GetParError(idx)
	return par, parError

def PlotRootDistributions(dataTree):
	gROOT.SetStyle('ATLAS')

	dataTree.Draw('psi_gen_rec>>hGen')
	dataTree.Draw('psi_truth_rec>>hTruth')
	hGen = gDirectory.Get('hGen')
	hGen.SetLineColor(kBlue)
	hGen.SetFillColor(kBlue)
	hTruth = gDirectory.Get('hTruth')
	hTruth.SetFillColor(kBlue)
	hTruth.SetLineColor(kBlue)

	hGen.GetXaxis().SetTitle('\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}[rad]')
	hGen.GetYaxis().SetTitle('Counts')
	hGen.GetYaxis().SetTitleOffset(1.7)
	hTruth.GetXaxis().SetTitle('\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}[rad]')
	hTruth.GetYaxis().SetTitle('Counts')
	hTruth.GetYaxis().SetTitleOffset(1.7)

	hGen = GaussianFit(hGen)
	hTruth = GaussianFit(hTruth)

	return hGen, hTruth

def PlotRootNeutronDependence(dataTree, groupLabels, ptLabels, parameter):
	gROOT.SetStyle('ATLAS')
	bins = 100

	#arrays are to be organized with x pertaining to a certain bin, then all sigmas pertaining
	x = array('f',[])
	ex =array('f',[])
	y_gen = array('f',[])
	ey_gen =array('f',[])
	y_truth = array('f',[])
	ey_truth =array('f',[])

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
			#input()
			dataTree.Draw('psi_gen_rec >> genTemp', cut)
			dataTree.Draw('psi_truth_rec >> truthTemp', cut)
			
			genTemp = gDirectory.Get('genTemp')
			truthTemp = gDirectory.Get('truthTemp')
			genTemp.Draw()

			genSigma, genError = GaussianFitGet(genTemp,parameter)
			truthSigma, truthError = GaussianFitGet(truthTemp, parameter)
			print(genSigma, truthSigma)

			y_gen.append(genSigma)
			ey_gen.append(genError)
			y_truth.append(truthSigma)
			ey_truth.append(truthError)

	genGraph = TMultiGraph()
	truthGraph = TMultiGraph()
	for i in range(len(ptLabels)):
		color = colors[i]
		tge_gen = TGraphErrors(n, x[i*4:i*4+4], y_gen[i*4:i*4+4], ex[i*4:i*4+4],ey_gen[i*4:i*4+4])
		tge_gen.SetDrawOption('AP')
		tge_truth = TGraphErrors(n, x[i*4:i*4+4], y_truth[i*4:i*4+4], ex[i*4:i*4+4],ey_truth[i*4:i*4+4])
		tge_truth.SetDrawOption('AP')

		lowPt = i*10 + 5
		highPt = i*10 + 15
		tge_gen.SetMarkerColor(color)
		tge_truth.SetMarkerColor(color)
		if highPt < 55:
			tge_gen.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}<' + str(highPt))
			tge_truth.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}<' + str(highPt))
		else:
			tge_gen.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}')
			tge_truth.SetTitle(str(lowPt)+'\leq p_{T}^{nuc}')
		tge_gen.SetLineColor(color)
		tge_truth.SetLineColor(color)
		tge_gen.SetMarkerStyle(21)
		tge_truth.SetMarkerStyle(21)
		genGraph.Add(tge_gen)
		truthGraph.Add(tge_truth)
	
	genGraph.SetTitle(';N_{neutrons};\sigma_{\Psi_{0}^{Gen-A}-\Psi_{0}^{Rec-A}} [rad]')
	truthGraph.SetTitle(';N_{neutrons};\sigma_{\Psi_{0}^{Truth-A}-\Psi_{0}^{Rec-A}} [rad]')
	genGraph.GetXaxis().SetLimits(20.,40.)
	genGraph.GetYaxis().SetRangeUser(0,1.4)
	genGraph.GetYaxis().SetLimits(0,1.4)
	truthGraph.GetXaxis().SetLimits(20.,40.)
	truthGraph.GetYaxis().SetRangeUser(0,2.5)
	truthGraph.GetYaxis().SetLimits(0,2.5)
	return genGraph, truthGraph

def RootPlot():
	model_loss = 'mse'
	file_num = 13
	filepath = f"/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/bdt_models/model_{file_num}_{model_loss}/"
	bins = 100

	#loads datasets
	test_A = pd.read_pickle(filepath + 'test_A.pickle')
	#set test_x based on model: 8:24 for allchan, 24:32 for avg, 24:26 for CoM
	test_X = test_A.iloc[:,8:24]
	Q_avg = test_A.iloc[:,0:2]
	psi_truth = test_A.iloc[:,5]
	pt_nuc = test_A.iloc[:,4].multiply(1000)
	numParticles = test_A.iloc[:,7]
	
	modelX = xgb.XGBRegressor()
	modelX.load_model(filepath+f'bdtX_{file_num}_{model_loss}.json')
	modelY = xgb.XGBRegressor()
	modelY.load_model(filepath+f'bdtY_{file_num}_{model_loss}.json')

	recX = modelX.predict(test_X.to_numpy(), iteration_range=(0, modelX.best_iteration))
	recY = modelY.predict(test_X.to_numpy(), iteration_range=(0, modelY.best_iteration))
	psi_rec = np.arctan2(recY, recX)
	psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])

	print(psi_gen)
	print(psi_rec)

	#Using pandas df output
	psi_gen_rec = psi_gen.subtract(psi_rec)
	psi_gen_rec[psi_gen_rec > np.pi] = -2*np.pi + psi_gen_rec
	psi_gen_rec[psi_gen_rec < -np.pi] = 2*np.pi + psi_gen_rec

	psi_truth_rec = psi_truth.subtract(psi_rec)
	psi_truth_rec[psi_truth_rec > np.pi] = -2*np.pi + psi_truth_rec
	psi_truth_rec[psi_truth_rec < -np.pi] = 2*np.pi + psi_truth_rec

	treeA = array2tree(numParticles.to_numpy(dtype = [('numParticles',np.int32)]))
	array2tree(pt_nuc.to_numpy(dtype = [('pt_nuclear',np.float64)]), tree = treeA)
	array2tree(psi_gen_rec.to_numpy(dtype = [('psi_gen_rec',np.float64)]), tree = treeA)
	array2tree(psi_truth_rec.to_numpy(dtype=[('psi_truth_rec',np.float64)]), tree = treeA)

	#Plots histograms for gen-reco, truth-reco distributions

	hGen, hTruth = PlotRootDistributions(treeA)

	groupLabels = [22, 27, 32, 37]
	ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
	nGen, nTruth = PlotRootNeutronDependence(treeA, groupLabels, ptLabels, 2)

	w = 750
	h = 600
	c1 = TCanvas('c1','c1', w, h)
	c2 = TCanvas('c2','c2', w, h)
	c3 = TCanvas('c3','c3', w, h)
	c4 = TCanvas('c4','c4', w, h)

	l1 = TLine()
	l1.SetLineStyle(kDashed)
	l1.SetLineColor(kBlack)

	c1.cd()
	hGen.Draw()
	c2.cd()
	hTruth.Draw()
	c3.cd()
	nGen.Draw('AP')
	c3.BuildLegend().SetBorderSize(0)
	for i in range(len(groupLabels)-1):
		l1.DrawLine(25 + i*5, nGen.GetYaxis().GetXmin(),25+i*5, nGen.GetYaxis().GetXmax())
	c4.cd()
	nTruth.Draw('AP')
	c4.BuildLegend().SetBorderSize(0)
	for i in range(len(groupLabels)-1):
		l1.DrawLine(25 + i*5, nTruth.GetYaxis().GetXmin(),25+i*5, nTruth.GetYaxis().GetXmax())

	c1.cd()
	c1.SetFillColor(kWhite)
	c1.SaveAs(filepath + 'genDistribution.png')
	c2.cd()
	c2.SetFillColor(kWhite)
	c2.SaveAs(filepath + 'truthDistribution.png')
	c3.cd()
	c3.SetFillColor(kWhite)
	c3.SaveAs(filepath + 'genNeutronDependence.png')
	c4.cd()
	c4.SetFillColor(kWhite)
	c4.SaveAs(filepath + 'truthNeutronDependence.png')

