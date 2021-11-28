

void testTimeComp() {

  Double_t n_params_linear = 34;
  Double_t n_params_cnn = 91722;
  Double_t n_params_fcn = 206338;
  const int n = 3;
  /*
  Double_t n_samples[6]={1, 10, 100, 1000, 10000, 100000};
  Double_t time_cnn[6] = {0.164, 0.174, 0.191, 0.249, 0.485, 2.930};
  Double_t time_fcn[6] = {0.174, 0.208, 0.214, 0.229, 0.362, 1.823};
  //  Double_t time_linear[] = {0.963,0.243,0.161,0.1615,0.145,0.127};
  Double_t time_linear[6] = {0.127, 0.145, 0.16, 0.161, 0.243, 0.963};
  */
  Double_t n_samples[n]={100, 1000, 10000};
  Double_t time_cnn[n] = {0.191, 0.249, 0.485};
  Double_t time_fcn[n] = {0.214, 0.229, 0.362};
  //  Double_t time_linear[] = {0.963,0.243,0.161,0.1615,0.145,0.127};
  Double_t time_linear[n] = {0.16, 0.161, 0.243};


  TGraph *g_fcn = new TGraph(n,n_samples,time_fcn);
  TGraph *g_linear = new TGraph(n,n_samples,time_linear);
  TGraph *g_cnn = new TGraph(n,n_samples,time_cnn);
  
  TCanvas *c1 = new TCanvas("c1","c1",700,550);
    c1->SetLogx();
  c1->SetLogy();

  TMultiGraph *mg = new TMultiGraph();

  g_linear->SetMarkerColor(kBlack);
  g_linear->SetMarkerStyle(20);
  mg->Add(g_linear, "P");
  
  g_cnn->SetMarkerColor(kRed);
  g_cnn->SetMarkerStyle(21);
  mg->Add(g_cnn);
  

  g_fcn->SetMarkerColor(kBlue);
  g_fcn->SetMarkerStyle(22);
  mg->Add(g_fcn,"P");
  mg->GetXaxis()->SetTitle("# of test samples");
  mg->GetYaxis()->SetTitle("Time [s]");
  //  mg->GetXaxis()->SetRangeUser(1,100000);
  //  mg->GetYaxis()->SetRangeUser(0.01,4.);
  for (int i = 0; i < 6; ++i){
    std::cout << " linear i " << i << " x " << g_linear->GetPointX(i) <<  " y " << g_linear->GetPointY(i) << std::endl;
  }
      for (int i = 0; i < 6; ++i){
    std::cout << " fcn i " << i << " x " << g_fcn->GetPointX(i) <<  " y " << g_fcn->GetPointY(i) << std::endl;
  }
    for (int i = 0; i < 6; ++i){
    std::cout << " cnn i " << i << " x " << g_cnn->GetPointX(i) <<  " y " << g_cnn->GetPointY(i) << std::endl;
  }
  mg->Draw("ap");
  
  TLegend *l = new TLegend(0.22, 0.65, 0.35, 0.9);
  l->AddEntry(g_linear,"Linear","p");
  l->AddEntry(g_fcn,"FCN","p");
  l->AddEntry(g_cnn,"CNN","p");
  l->SetBorderSize(0);
  l->SetTextFont(43);
  l->SetTextSize(21);
  l->SetFillColor(0);
  l->Draw();



  

 }
