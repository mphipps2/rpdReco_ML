void test() {
  
  Double_t time_cnn[6] = { 0.164, 0.174, 0.191, 0.249, 0.485, 2.930};
  Double_t time_fcn[6] = { 0.174, 0.208, 0.214, 0.229, 0.362, 1.823};
  Double_t time_linear[6] = {0.127, 0.145, 0.16, 0.161, 0.243, 0.963};
  Double_t time_bdt[6] = {0.0163, 0.0769, 0.0304, 0.1184, 0.3133, 2.1428};
  Double_t x[6] = {1, 10, 100, 1000, 10000, 100000};
  TCanvas *c1 = new TCanvas("c1","c1",550,500);
  TGraph *g_cnn = new TGraph(6, x, time_cnn);
  g_cnn->SetMarkerColor(kRed);
  TGraph *g_fcn = new TGraph(6, x, time_fcn);
  g_fcn->SetMarkerColor(kBlue);
  TGraph *g_linear = new TGraph(6, x, time_linear);
  g_linear->SetMarkerColor(kBlack);
  TGraph *g_bdt = new TGraph(6, x, time_bdt);
  g_bdt->SetMarkerColor(8);
  /*
  c1->Divide(2,2);


  c1->cd(1);
  gPad->SetLogx();
  gPad->SetLogy();
  g_cnn->Draw("AP");
  
  c1->cd(2);
  gPad->SetLogx();
  gPad->SetLogy();
  g_fcn->Draw("AP");
  */
  TMultiGraph *mg1 = new TMultiGraph("mg1","mg1");
  c1->cd(1);
  
  gPad->SetLogx();
  gPad->SetLogy();
  
  mg1->Add(g_cnn);
  mg1->Add(g_fcn);
  mg1->Add(g_bdt);

  mg1->Add(g_linear);
  mg1->Draw("AP");
  mg1->GetXaxis()->SetTitle("# of Test Samples");
  mg1->GetYaxis()->SetTitle("Time [s]");

  TLegend *l = new TLegend(0.22, 0.65, 0.35, 0.9);
  l->AddEntry(g_linear,"Linear","p");
  l->AddEntry(g_fcn,"FCN","p");
  l->AddEntry(g_cnn,"CNN","p");
  l->AddEntry(g_bdt,"BDT","p");
  l->SetBorderSize(0);
  l->SetTextFont(43);
  l->SetTextSize(21);
  l->SetFillColor(0);
  l->Draw();
  
  c1->SaveAs(Form("/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Training/ToyFermi_qqFibers_LHC_noPedNoise/TestTimeComp.png"));
  
  
}
