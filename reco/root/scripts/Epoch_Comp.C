#include <iostream>

void Epoch_Comp() {

  gROOT->SetStyle("ATLAS");

  Double_t epochs_cnn[8] = { 170, 135, 82, 67, 53, 74, 93, 60};
  Double_t epochs_fcn[8] = { 102, 77, 49, 42, 36, 40, 37, 45};
  Double_t epochs_bdtX[8] = { 100, 96, 97, 92, 100, 93, 95, 88};
  Double_t epochs_bdtY[8] = { 100, 97, 100, 99, 96, 98, 92, 94};
  Double_t epochs_lin[8] = { 500, 77, 49, 42, 36, 40, 37, 45};
  Double_t n_training_samples[8] = {6250, 12500, 25000, 50000, 100000, 200000, 400000, 800000};
  TGraph *g_cnn = new TGraph(8, n_training_samples, epochs_cnn);
  TGraph *g_fcn = new TGraph(8, n_training_samples, epochs_fcn);
  TGraph *g_bdtX = new TGraph(8, n_training_samples, epochs_bdtX);
  TGraph *g_bdtY = new TGraph(8, n_training_samples, epochs_bdtY);
  TGraph *g_lin = new TGraph(8, n_training_samples, epochs_lin);

  TMultiGraph *mg1 = new TMultiGraph("mg1","mg1");

  TCanvas *c2 = new TCanvas("c2","c2",550,500);
  g_cnn->SetMarkerColor(kRed);
  g_fcn->SetMarkerColor(kBlack);
  g_bdtX->SetMarkerColor(kBlue);
  g_bdtY->SetMarkerColor(kBlue);
  g_bdtY->SetMarkerStyle(kCircle);
  g_lin->SetMarkerColor(kGreen+2);
  //  gStyle->SetPalette(kVisibleSpectrum);
  gPad->SetRightMargin(0.08);
  gPad->SetLeftMargin(0.18);
  mg1->Add(g_cnn,"P");
  mg1->Add(g_fcn,"P");
  mg1->Add(g_bdtX,"P");
  mg1->Add(g_bdtY,"P");
  mg1->Add(g_lin,"P");

  mg1->GetXaxis()->SetTitle("# of Training Samples");

  mg1->GetYaxis()->SetTitle("Epochs");
  mg1->GetYaxis()->SetTitleOffset(1.8);
  mg1->GetXaxis()->SetNdivisions(504);
  mg1->GetXaxis()->SetLimits(3000,1000000);
  c2->SetLogx();
  //  mg1->SetMaximum(5000000); mg1->SetMinimum(1);
  mg1->GetYaxis()->SetRangeUser(20,201);
  mg1->Draw("APE");
  //  mg1->GetXaxis()->SetRangeUser(0,5);

  TLegend *leg = new TLegend(0.54,0.6,0.9,0.88);
  leg->SetBorderSize(0);
  leg->SetTextFont(43);
  leg->SetTextSize(21);
  leg->SetFillColor(0);
  leg->SetHeader("Fibers: qq","C");
  leg->AddEntry(g_cnn,"CNN 2d","plfe");
  leg->AddEntry(g_fcn,"FCN 2d","plfe");
  leg->AddEntry(g_lin,"Linear 2d","plfe");
  leg->AddEntry(g_bdtX,"BDT 1d Q_{x}","plfe");
  leg->AddEntry(g_bdtY,"BDT 1d Q_{y}","plfe");
  leg->Draw();

  c2->SaveAs(Form("/mnt/c/Users/Fre Shava Cado/Documents/VSCode Projects/SaveFiles/ToyFermi_qqFibers_LHC_noPedNoise/EpochComp_TrainingSize_allModels.png"));

  delete g_cnn; delete g_fcn; delete g_bdtX; delete g_bdtY; delete g_lin; 
  delete c2;
}