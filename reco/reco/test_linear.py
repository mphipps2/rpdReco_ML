import sys

from tensorflow.python.ops.control_flow_ops import group
sys.path.append('/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/rpdreco/reco')

#import reco.lib.norm as norm
import reco.lib.models as models
import reco.lib.process as process
#import reco.lib.vis as vis
import reco.lib.io as io

import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from array import array

from ROOT import *
from root_numpy import array2tree

def fit_function(x, A, mu, sigma):
	return A*np.exp(-(x-mu)**2/(2*sigma**2))
	
def fit_function2(x, A, B, mu, sigma):
	return A + B*np.exp(-1*((x-mu)**2/(2*sigma**2)))

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

def PlotMplDistributions(measuredDf, filepath, file_num, df, groupLabels, ptLabels):
	bins = 100

    	#collects summary stats for difference in angle
	mean_gen, sigma_gen = norm.fit(measuredDf.psi_gen)
	error_gen = measuredDf.psi_gen.sem()

	mean_truth, sigma_truth = norm.fit(measuredDf.psi_truth)
	error_truth = np.round(measuredDf.psi_truth.sem(), 3)

	genBins = np.linspace(mean_gen-sigma_gen, mean_gen+sigma_gen, bins + 1)
	genBins2 = np.linspace(measuredDf.psi_gen.min(), measuredDf.psi_gen.max(), bins + 1)
	genEntries, bins_1 = np.histogram(measuredDf.psi_gen, bins = genBins, density = False)
	genEntries2, bins_2 = np.histogram(measuredDf.psi_gen, bins = genBins2, density = False)
	genCenters = np.array([0.5*(genBins[i]+genBins[i+1]) for i in range(len(genBins)-1)])
	genCenters2 = np.array([0.5*(genBins2[i]+genBins2[i+1]) for i in range(len(genBins2)-1)])
	genPopt, genPcov = curve_fit(fit_function, xdata = genCenters, ydata = genEntries, p0 = [75000, mean_gen, sigma_gen])
	genPopt1, genPcov2 = curve_fit(fit_function2, xdata = genCenters2, ydata = genEntries2, p0 = [60000, 75000, mean_gen, sigma_gen])
	genXspace = np.linspace(mean_gen-sigma_gen, mean_gen+sigma_gen, 1000000)

	truthBins = np.linspace(mean_truth-sigma_truth, mean_truth+sigma_truth, bins + 1)
	truthBins2 = np.linspace(measuredDf.psi_truth.min(), measuredDf.psi_truth.max(), bins + 1)
	truthEntries, bins_3 = np.histogram(measuredDf.psi_truth, bins = truthBins, density = False)
	truthEntries2, bins_4 = np.histogram(measuredDf.psi_truth, bins = truthBins2, density = False)
	truthCenters = np.array([0.5*(truthBins[i]+truthBins[i+1]) for i in range(len(truthBins)-1)])
	truthCenters2 = np.array([0.5*(truthBins2[i]+truthBins2[i+1]) for i in range(len(truthBins2)-1)])
	truthPopt, truthPcov = curve_fit(fit_function, xdata = truthCenters, ydata = truthEntries, p0 = [75000, mean_truth, sigma_truth])
	truthPopt1, truthPcov2 = curve_fit(fit_function2, xdata = truthCenters2, ydata = truthEntries2, p0 = [60000, 75000, mean_truth, sigma_truth])
	truthXspace = np.linspace(mean_truth-sigma_truth, mean_truth+sigma_truth, 1000000)

	title = r'$45<p_{T}^{nuclear}$ MeV'

	#plots difference in angle
	plt.figure(2)
	plt.hist(measuredDf.psi_gen, bins = bins, density = False)
	plt.plot(genXspace, fit_function2(genXspace,*genPopt1))
	plt.ylim(bottom = 0)
	#plt.title(title+' Gen', fontsize = 12)
	plt.xlabel(r'$\Psi_0^{\rm Gen-A}-\Psi_0^{\rm Rec-A}$ [rad]', fontsize = 12)
	plt.ylabel('Density Function', fontsize = 12)
	#plt.ylim(top = 0.21)
	plt.text(-3,200,f'$\\mu={np.round(mean_gen, 3)}\\pm {np.round(error_gen,3)}$,\n $\\sigma={np.round(sigma_gen, 3)}$')
	plt.savefig(filepath + f'//model{file_num}_gen_anglediff.png')

	plt.figure(3)
	plt.hist(measuredDf.psi_truth, bins = bins, density = False)
	plt.plot(truthXspace, fit_function2(truthXspace, *truthPopt1))
	#plt.title(title+' Truth', fontsize = 12)
	plt.xlabel(r'$\Psi_0^{\rm Truth-A}-\Psi_0^{\rm Rec-A}$ [rad]', fontsize = 12)
	plt.ylabel('Density Function', fontsize = 12)
	#plt.ylim(top =0.21)
	plt.text(-3,200,f'$\\mu={np.round(mean_truth, 3)}\\pm {np.round(error_truth,3)}$,\n $\\sigma={np.round(sigma_truth, 3)}$')
	plt.savefig(filepath + f'//model{file_num}_truth_anglediff.png')

def GetMplNeutronDependence(df, ptLabels, groupLabels):
	bins = 100
	
	sem = pd.DataFrame(0, index = pd.MultiIndex.from_product([ptLabels,groupLabels], names = ['ptBins','nbins']), columns = ['sigma_gen','sigma_truth'])
	std = pd.DataFrame(0, index = pd.MultiIndex.from_product([ptLabels,groupLabels], names = ['ptBins','nbins']), columns = ['sigma_gen','sigma_truth'])
	for pt in ptLabels:
		for nCount in groupLabels:
			testdf = df.loc[(df['ptBins'] == pt) & (df['nbins'] == nCount)]

			mean_test_gen, sigma_test_gen = norm.fit(testdf.psi_gen)
			testBins_gen = np.linspace(mean_test_gen-sigma_test_gen, mean_test_gen+sigma_test_gen, bins+1)
			genEntries, bins_1 = np.histogram(testdf.psi_gen, bins = testBins_gen, density = False)
			genCenters = np.array([0.5*(testBins_gen[i]+testBins_gen[i+1]) for i in range (len(testBins_gen)-1)])
			#genPopt, genPcov = curve_fit(fit_function, xdata = genCenters, ydata = genEntries, p0 = [500, mean_test_gen, sigma_test_gen])

			mean_test_truth, sigma_test_truth = norm.fit(testdf.psi_truth)
			testBins_truth = np.linspace(mean_test_truth-sigma_test_truth, mean_test_truth+sigma_test_truth, bins+1)
			truthEntries, bins_2 = np.histogram(testdf.psi_truth, bins = testBins_truth, density = False)
			truthCenters = np.array([0.5*(testBins_truth[i]+testBins_truth[i+1]) for i in range (len(testBins_truth)-1)])
			#truthPopt, truthPcov = curve_fit(fit_function, xdata = truthCenters, ydata = truthEntries, p0 = [500, mean_test_truth, sigma_test_truth])
			#std.loc[(pt,nCount)] += [np.abs(genPopt[-1]), np.abs(truthPopt[-1])]
			#sem.loc[(pt,nCount)] += [np.sqrt(np.diag(genPcov))[-1], np.sqrt(np.diag(truthPcov))[-1]]
			std.loc[(pt,nCount)] += [testdf.psi_gen.std(), testdf.psi_truth.std()]
			sem.loc[(pt,nCount)] += [testdf.psi_gen.sem(), testdf.psi_truth.sem()]
	return std, sem 

def PlotMplNeutronDependence(std, sem, groupLabels, filepath, file_num):
	plt.figure(4)
	ax = plt.figure(4).gca()
	ax.xaxis.set_major_locator(MaxNLocator(nbins = 4, integer=True))
	plt.plot(groupLabels, std.loc['pt0'].sigma_gen, 'gs', label = r'$5\leq p_T^{nuclear}<15$ MeV')
	plt.errorbar(groupLabels, std.loc['pt0'].sigma_gen, yerr=sem.loc['pt0'].sigma_gen, fmt = 'gs')
	plt.plot(groupLabels, std.loc['pt1'].sigma_gen, 'bs', label = r'$15\leq p_T^{nuclear}<25$ MeV')
	plt.errorbar(groupLabels, std.loc['pt1'].sigma_gen, yerr=sem.loc['pt1'].sigma_gen, fmt = 'bs')
	plt.plot(groupLabels, std.loc['pt2'].sigma_gen, 'rs', label = r'$25\leq p_T^{nuclear}<35$ MeV')
	plt.errorbar(groupLabels, std.loc['pt2'].sigma_gen, yerr=sem.loc['pt2'].sigma_gen, fmt = 'rs')
	plt.plot(groupLabels, std.loc['pt3'].sigma_gen, 'ms', label = r'$35\leq p_T^{nuclear}<45$ MeV')
	plt.errorbar(groupLabels, std.loc['pt3'].sigma_gen, yerr=sem.loc['pt3'].sigma_gen, fmt = 'ms')
	plt.plot(groupLabels, std.loc['pt4'].sigma_gen, 'ks', label = r'$45\leq p_T^{nuclear}$ MeV')
	plt.errorbar(groupLabels, std.loc['pt4'].sigma_gen, yerr=sem.loc['pt4'].sigma_gen, fmt = 'ks')
	plt.grid()
	#plt.title('Gen Resolution', fontsize = 12)
	plt.xlabel(r'$\rm N_{neutrons}$', fontsize = 12)
	plt.ylabel(r'$\sigma_{\rm \Psi^{\rm Gen-A}_0-\Psi^{Rec-A}_0}$ [rad]', fontsize = 12)
	plt.ylim(bottom = 0, top = 3)
	plt.legend()
	plt.savefig(filepath +  f'//model{file_num}_gen_stratsigmas.png')
    
	plt.figure(5)
	ax = plt.figure(5).gca()
	ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, integer=True))
	plt.plot(groupLabels, std.loc['pt0'].sigma_truth, 'gs', label = r'$5\leq p_T^{nuclear}<15$ MeV')
	plt.errorbar(groupLabels, std.loc['pt0'].sigma_truth, yerr=sem.loc['pt0'].sigma_truth, fmt = 'gs')
	plt.plot(groupLabels, std.loc['pt1'].sigma_truth, 'bs', label = r'$15\leq p_T^{nuclear}<25$ MeV')
	plt.errorbar(groupLabels, std.loc['pt1'].sigma_truth, yerr=sem.loc['pt1'].sigma_truth, fmt = 'bs')
	plt.plot(groupLabels, std.loc['pt2'].sigma_truth, 'rs', label = r'$25\leq p_T^{nuclear}<35$ MeV')
	plt.errorbar(groupLabels, std.loc['pt2'].sigma_truth, yerr=sem.loc['pt2'].sigma_truth, fmt = 'rs')
	plt.plot(groupLabels, std.loc['pt3'].sigma_truth, 'ms', label = r'$35\leq p_T^{nuclear}<45$ MeV')
	plt.errorbar(groupLabels, std.loc['pt3'].sigma_truth, yerr=sem.loc['pt3'].sigma_truth, fmt = 'ms')
	plt.plot(groupLabels, std.loc['pt4'].sigma_truth, 'ks', label = r'$45\leq p_T^{nuclear}$ MeV')
	plt.errorbar(groupLabels, std.loc['pt4'].sigma_truth, yerr=sem.loc['pt4'].sigma_truth, fmt = 'ks')
	plt.grid()
	#plt.title('Truth Resolution')
	plt.xlabel(r'$\rm N_{\rm neutrons}$', fontsize = 12)
	plt.ylabel(r'$\sigma_{\rm \Psi^{\rm Truth-A}_0-\Psi^{Rec-A}_0}$ [rad]')
	plt.ylim(bottom = 0, top = 6)
	plt.legend()
	plt.savefig(filepath + f'//model{file_num}_truth_stratsigmas.png')

def MplPlot():
	model_loss = 'mse'
	file_num = 5
	filepath = f"/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/models/model_{file_num}_{model_loss}"
	bins = 100

	#loads datasets
	test_A = pd.read_pickle(filepath + '//test_A.pickle')
	#set test_x based on model: 8:24 for allchan, 24:32 for avg, 24:26 for CoM
	test_X = test_A.iloc[:,8:24]
	Q_avg = test_A.iloc[:,0:2]
	psi_truth = test_A.iloc[:,5]
	pt_nuc = test_A.iloc[:,4]
	numParticles = test_A.iloc[:,7]

	'''
	test_X = pd.read_pickle(filepath + '//test_X.pickle')
	Q_avg = pd.read_pickle(filepath + '//test_y.pickle')
	psi_truth = pd.read_pickle(filepath + '//psi_truth.pickle')
 	'''
	
	model = keras.models.load_model(filepath+f'//linear_{file_num}_{model_loss}.h5',compile = False)
	Q_predicted = model.predict([test_X.astype('float')])

	f = open(filepath + f'linear_{file_num}_{model_loss}_summary.txt', 'w')
	model.summary(print_fn = lambda x: f.write(x+'\n'))
	f.close()

	psi_rec = np.arctan2(Q_predicted[:,1],Q_predicted[:,0])
	psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])

	psi_gen_rec = psi_gen.subtract(psi_rec)
	psi_gen_rec[psi_gen_rec > np.pi] = -2*np.pi + psi_gen_rec
	psi_gen_rec[psi_gen_rec < -np.pi] = 2*np.pi + psi_gen_rec

	psi_truth_rec = psi_truth.subtract(psi_rec)
	psi_truth_rec[psi_truth_rec > np.pi] = -2*np.pi + psi_truth_rec
	psi_truth_rec[psi_truth_rec < -np.pi] = 2*np.pi + psi_truth_rec

	df = pd.DataFrame()
	df['numParticles'] = numParticles
	df['pt_nuclear'] = pt_nuc.multiply(1000)
	df['psi_gen'] = psi_gen_rec
	df['psi_truth'] = psi_truth_rec

	#stratifies based on pt_nuclear
	ptLabels = ['pt0', 'pt1', 'pt2', 'pt3', 'pt4']
	df['ptBins'] = pd.cut(x = df.iloc[:,1], bins = [5, 15, 25, 35, 45, np.inf], labels = ptLabels, right = False, include_lowest = True)
	#creates bins for neutron clusters. For future reference, do not do this in python
	groupLabels = [22, 27, 32, 37]
	df['nbins'] = pd.cut(x = df.iloc[:,0], bins =[20,25,30,35,40], labels = groupLabels, right = False, include_lowest = True)

	measuredDf = df #df[df['ptBins'] == 'pt4']

	PlotMplDistributions(measuredDf, filepath, file_num, df, groupLabels, ptLabels)

	std, sem = GetMplNeutronDependence(df, ptLabels, groupLabels)
	PlotMplNeutronDependence(std, sem, groupLabels, filepath, file_num)

	plt.show()


def RootPlot():
	model_loss = 'mse'
	file_num = 28
	filepath = f"/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/models/model_{file_num}_{model_loss}/"
	bins = 100

	#loads datasets
	test_A = pd.read_pickle(filepath + 'test_A.pickle')
	#set test_x based on model: 8:24 for allchan, 24:32 for avg, 24:26 for CoM
	test_X = test_A.iloc[:,8:24]
	Q_avg = test_A.iloc[:,0:2]
	psi_truth = test_A.iloc[:,5]
	pt_nuc = test_A.iloc[:,4].multiply(1000)
	numParticles = test_A.iloc[:,7]
	
	model = keras.models.load_model(filepath+f'linear_{file_num}_{model_loss}.h5',compile = False)
	Q_predicted = model.predict([test_X.astype('float')])
	
	psi_rec = np.arctan2(Q_predicted[:,1],Q_predicted[:,0])
	psi_gen = np.arctan2(Q_avg.iloc[:,1],Q_avg.iloc[:,0])

	psi_gen_rec = psi_gen.subtract(psi_rec)
	psi_gen_rec[psi_gen_rec > np.pi] = -2*np.pi + psi_gen_rec
	psi_gen_rec[psi_gen_rec < -np.pi] = 2*np.pi + psi_gen_rec

	psi_truth_rec = psi_truth.subtract(psi_rec)
	psi_truth_rec[psi_truth_rec > np.pi] = -2*np.pi + psi_truth_rec
	psi_truth_rec[psi_truth_rec < -np.pi] = 2*np.pi + psi_truth_rec

	treeA = array2tree(numParticles.to_numpy(dtype = [('numParticles',np.int32)]))
	array2tree(pt_nuc.to_numpy(dtype = [('pt_nuclear',np.float64)]), tree = treeA)
	array2tree(psi_gen_rec.to_numpy(dtype = [('psi_gen_rec',np.float64)]), tree = treeA)
	array2tree(psi_truth_rec.to_numpy(dtype = [('psi_truth_rec',np.float64)]), tree = treeA)

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
	



	