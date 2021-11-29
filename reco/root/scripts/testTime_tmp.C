

void testTimeComp() {
  Double_t loss_cnn[6] = { 0.164, 0.174, 0.191, 0.249, 0.485, 2.930};
  Double_t loss_fcn[6] = { 0.174, 0.208, 0.214, 0.229, 0.362, 1.823};
  Double_t ex[6] = {0.,0.,0.,0.,0.,0.};
  Double_t ey[6] = {0.,0.,0.,0.,0.,0.};
  Double_t n_training_samples[6] = {1, 10, 100, 1000, 10000, 100000};
  TGraphErrors *g_cnn = new TGraphErrors(6, n_training_samples, loss_cnn, ex, ey);

  TCanvas *c2 = new TCanvas("c2","c2",550,500);
  g_cnn->SetMinimum(0.09);
  //  g_cnn->Draw("AP");
  
  TGraphErrors *g_fcn = new TGraphErrors(6, n_training_samples, loss_fcn, ex, ey);
  g_fcn->SetMinimum(0.09);
  //  g_fcn->Draw("AP");
  
  TMultiGraph *mg1 = new TMultiGraph("mg1","mg1");


  g_cnn->SetMarkerColor(kRed);
  g_fcn->SetMarkerColor(kBlack);

  gPad->SetRightMargin(0.15);
  mg1->Add(g_cnn,"P");
  mg1->Add(g_fcn,"P");

  //  mg1->GetXaxis()->SetTitle("# of Training Samples");

  //  mg1->GetYaxis()->SetTitle("Mean Squared Error");

  c2->SetLogy();
  //  c2->SetLogx();
  //  mg1->SetMaximum(5000000); mg1->SetMinimum(1);
  //  mg1->GetYaxis()->SetRangeUser(0.,0.7);
  mg1->SetMinimum(0.09);
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
  
}
