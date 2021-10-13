#include <iostream>

void Loss_Comp() {

  Double_t loss_cnn[6] = { 0.265826, 0.265184, 0.26518, 0.26146, 0.260608, 0.2601078};
  Double_t loss_fcn[6] = { 0.3066, 0.2851, 0.2751, 0.2705, 0.2650, 0.2631};
  Double_t n_training_samples[6] = {25000, 50000, 100000, 200000, 400000, 800000};
  TGraph *g_cnn = new TGraph(6, n_training_samples, loss_cnn);
  TGraph *g_fcn = new TGraph(6, n_training_samples, loss_fcn);

  TMultiGraph *mg1 = new TMultiGraph("mg1","mg1");

  TCanvas *c2 = new TCanvas("c2","c2",550,500);
  g_cnn->SetMarkerColor(kRed);
  g_fcn->SetMarkerColor(kBlack);
  //  gStyle->SetPalette(kVisibleSpectrum);
  gPad->SetRightMargin(0.15);
  mg1->Add(g_cnn,"P");
  mg1->Add(g_fcn,"P");

  mg1->GetXaxis()->SetTitle("# of Training Samples");

  mg1->GetYaxis()->SetTitle("Mean Squared Error");
  mg1->GetXaxis()->SetNdivisions(504);
  c2->SetLogx();
  //  mg1->SetMaximum(5000000); mg1->SetMinimum(1);
  //  mg1->GetYaxis()->SetRangeUser(0.,0.7);
  mg1->Draw("APE");
  //  mg1->GetXaxis()->SetRangeUser(0,5);

  TLegend *leg = new TLegend(0.5,0.71,0.8,0.93);
  leg->SetBorderSize(0);
  leg->SetTextFont(43);
  leg->SetTextSize(21);
  leg->SetFillColor(0);
  leg->AddEntry(g_cnn,"CNN Model","plfe");
  leg->AddEntry(g_fcn,"FCN Model","plfe");
  leg->Draw();

  TLatex *tex = new TLatex();
  tex->SetNDC();
  tex->SetTextFont(43);
  tex->SetTextSize(21);
  tex->SetLineWidth(2);
  tex->DrawLatex(0.21,0.8,"Fibers: qq");

  c2->SaveAs(Form("/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Training/ToyFermi_qqFibers_LHC_noPedNoise/LossComp_TrainingSize_cnnVsfcn.png"));

  delete g_cnn; delete g_fcn; 
  delete c2;
}
