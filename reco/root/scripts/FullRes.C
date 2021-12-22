void FullRes() {

  char output_dir[256] = "/mnt/c/Users/mwp89/Desktop/ZDC/RPD/ML_Testing/ToyFermi_array_qpFibers_LHC_noPedNoise/Model1/";
  Double_t upperRange = 1.01;
  Double_t res_k1pt4[4] = {0.810, 0.856, 0.884, 0.896};
  Double_t res_k1pt3[4] = {0.757, 0.794, 0.821, 0.851};
  Double_t res_k1pt2[4] = {0.653, 0.663, 0.724, 0.775};
  Double_t res_k1pt1[4] = {0.426, 0.483, 0.536, 0.569};
  Double_t res_k1pt0[4] = {0.200, 0.255, 0.322, 0.374};
  
  Double_t err_k1pt4[4] = {0.023, 0.017, 0.016, 0.018};
  Double_t err_k1pt3[4] = {0.020, 0.015, 0.014, 0.016};
  Double_t err_k1pt2[4] = {0.025, 0.021, 0.019, 0.020};
  Double_t err_k1pt1[4] = {0.043, 0.032, 0.030, 0.033};
  Double_t err_k1pt0[4] = {0.095, 0.064, 0.052, 0.052 };

  

  /*
  // rpd12_AB
  Double_t res_k1pt4[4] = {0.864, 0.893, 0.918, 0.930};
  Double_t res_k1pt3[4] = {0.812, 0.846, 0.878, 0.899};
  Double_t res_k1pt2[4] = {0.693, 0.726, 0.783, 0.825};
  Double_t res_k1pt1[4] = {0.525, 0.556, 0.620, 0.630};
  Double_t res_k1pt0[4] = {0.274, 0.315, 0.375, 0.444};

  Double_t err_k1pt4[4] = {0.020, 0.014, 0.012, 0.013};
  Double_t err_k1pt3[4] = {0.017, 0.0129, 0.011, 0.012};
  Double_t err_k1pt2[4] = {0.023, 0.018, 0.016, 0.017};
  Double_t err_k1pt1[4] = {0.034, 0.027, 0.025, 0.029};
  Double_t err_k1pt0[4] = {0.069, 0.050, 0.044, 0.043};
  */

  Double_t res_k2pt4[4] = {0.498,0.527,0.592,0.649};
  Double_t res_k2pt3[4] = {0.413,0.457,0.527,0.556};
  Double_t res_k2pt2[4] = {0.279,0.292,0.346,0.401};
  Double_t res_k2pt1[4] = {0.,0.078,0.183,0.208};
  Double_t res_k2pt0[4] = {0.106,0.,0.,0.};
  Double_t res_k3pt4[4] = {0.227,0.249,0.352,0.424};
  Double_t res_k3pt3[4] = {0.221,0.212,0.327,0.310};
  Double_t res_k3pt2[4] = {0.184,0.084,0.050,0.206};
  Double_t res_k3pt1[4] = {0.118,0.,0.093,0.109};
  Double_t res_k3pt0[4] = {0.151,0.,0.116,0.144};

  Double_t err_k2pt4[4] = {0.019,0.015,0.013,0.013};
  Double_t err_k2pt3[4] = {0.017,0.013,0.011, 0.012};
  Double_t err_k2pt2[4] = {0.027,0.021,0.018,0.018};
  Double_t err_k2pt1[4] = {0.,0.083,0.037,0.039};
  Double_t err_k2pt0[4] = {0.,0.064,0.,0.};
  Double_t err_k3pt4[4] = {0.058,0.044,0.030,0.027};
  Double_t err_k3pt3[4] = {0.042,0.037,0.023,0.029};
  Double_t err_k3pt2[4] = {0.052,0.106,0.20,0.048};
  Double_t err_k3pt1[4] = {0.084,0.,0.098,0.099};
  Double_t err_k3pt0[4] = {0.065,0.,0.08,0.07};


  
  Double_t x[4] = {22.5, 27.5, 32.5, 37.5};
  Double_t ex[4] = {0., 0., 0., 0.};
  Int_t n = 4;

  TCanvas *c1 = new TCanvas("c1","c1",750, 600);
  TMultiGraph *mg = new TMultiGraph();

  TGraphErrors *tge_k1pt0 = new TGraphErrors(n, x, res_k1pt0, ex, err_k1pt0);
  TGraphErrors *tge_k1pt1 = new TGraphErrors(n, x, res_k1pt1, ex, err_k1pt1);
  TGraphErrors *tge_k1pt2 = new TGraphErrors(n, x, res_k1pt2, ex, err_k1pt2);
  TGraphErrors *tge_k1pt3 = new TGraphErrors(n, x, res_k1pt3, ex, err_k1pt3);
  TGraphErrors *tge_k1pt4 = new TGraphErrors(n, x, res_k1pt4, ex, err_k1pt4);
  

  tge_k1pt0->SetTitle(Form("%d #leq p_{T}^{spec}< %d",5,15));              
  tge_k1pt1->SetTitle(Form("%d #leq p_{T}^{spec}< %d",15,25));              
  tge_k1pt2->SetTitle(Form("%d #leq p_{T}^{spec}< %d",25,35));              
  tge_k1pt3->SetTitle(Form("%d #leq p_{T}^{spec}< %d",35,45));              
  tge_k1pt4->SetTitle(Form("%d #leq p_{T}^{spec}",45));              

  tge_k1pt0->SetMarkerColor(kBlue);
  tge_k1pt1->SetMarkerColor(kCyan+1);
  tge_k1pt2->SetMarkerColor(kSpring);
  tge_k1pt3->SetMarkerColor(kOrange);
  tge_k1pt4->SetMarkerColor(kRed);
  tge_k1pt0->SetLineColor(kBlue);
  tge_k1pt1->SetLineColor(kCyan+1);
  tge_k1pt2->SetLineColor(kSpring);
  tge_k1pt3->SetLineColor(kOrange);
  tge_k1pt4->SetLineColor(kRed);
  tge_k1pt0->SetMarkerStyle(21);
  tge_k1pt1->SetMarkerStyle(21);
  tge_k1pt2->SetMarkerStyle(21);
  tge_k1pt3->SetMarkerStyle(21);
  tge_k1pt4->SetMarkerStyle(21);
  
  mg->Add(tge_k1pt0);
  mg->Add(tge_k1pt1);
  mg->Add(tge_k1pt2);
  mg->Add(tge_k1pt3);
  mg->Add(tge_k1pt4);
  
  mg->SetTitle(";N_{neutrons};Res(#Psi_{1})");

  mg->GetXaxis()->SetLimits(20.,40.);
  mg->GetYaxis()->SetRangeUser(0,upperRange);
  mg->GetYaxis()->SetLimits(0,upperRange);
   

    
  TLine *l1 = new TLine();
  l1->SetLineStyle(kDashed);
  l1->SetLineColor(kBlack);

  c1->cd();
  mg->Draw("APE");
  c1->BuildLegend(0.5,0.17,0.84,0.36)->SetBorderSize(0);
 
  for (int i = 0; i < 3; ++i) {
    l1->DrawLine(25 + i*5, mg->GetYaxis()->GetXmin(),25+i*5, mg->GetYaxis()->GetXmax());
  }

  c1->SetFillColor(kWhite);

  c1->SaveAs(Form("%s/NeutronDepCosResolution_FullRes_k1_rpd2_AB.png",output_dir));
 





    TCanvas *c2 = new TCanvas("c2","c2",750, 600);
  TMultiGraph *mg2 = new TMultiGraph();

  TGraphErrors *tge_k2pt0 = new TGraphErrors(n, x, res_k2pt0, ex, err_k2pt0);
  TGraphErrors *tge_k2pt1 = new TGraphErrors(n, x, res_k2pt1, ex, err_k2pt1);
  TGraphErrors *tge_k2pt2 = new TGraphErrors(n, x, res_k2pt2, ex, err_k2pt2);
  TGraphErrors *tge_k2pt3 = new TGraphErrors(n, x, res_k2pt3, ex, err_k2pt3);
  TGraphErrors *tge_k2pt4 = new TGraphErrors(n, x, res_k2pt4, ex, err_k2pt4);
  

  tge_k2pt0->SetTitle(Form("%d #leq p_{T}^{spec}< %d",5,15));              
  tge_k2pt1->SetTitle(Form("%d #leq p_{T}^{spec}< %d",15,25));              
  tge_k2pt2->SetTitle(Form("%d #leq p_{T}^{spec}< %d",25,35));              
  tge_k2pt3->SetTitle(Form("%d #leq p_{T}^{spec}< %d",35,45));              
  tge_k2pt4->SetTitle(Form("%d #leq p_{T}^{spec}",45));              

  tge_k2pt0->SetMarkerColor(kBlue);
  tge_k2pt1->SetMarkerColor(kCyan+1);
  tge_k2pt2->SetMarkerColor(kSpring);
  tge_k2pt3->SetMarkerColor(kOrange);
  tge_k2pt4->SetMarkerColor(kRed);
  tge_k2pt0->SetLineColor(kBlue);
  tge_k2pt1->SetLineColor(kCyan+1);
  tge_k2pt2->SetLineColor(kSpring);
  tge_k2pt3->SetLineColor(kOrange);
  tge_k2pt4->SetLineColor(kRed);
  tge_k2pt0->SetMarkerStyle(21);
  tge_k2pt1->SetMarkerStyle(21);
  tge_k2pt2->SetMarkerStyle(21);
  tge_k2pt3->SetMarkerStyle(21);
  tge_k2pt4->SetMarkerStyle(21);
  
  mg2->Add(tge_k2pt0);
  mg2->Add(tge_k2pt1);
  mg2->Add(tge_k2pt2);
  mg2->Add(tge_k2pt3);
  mg2->Add(tge_k2pt4);
  
  mg2->SetTitle(";N_{neutrons};Res(2#Psi_{1})");

  mg2->GetXaxis()->SetLimits(20.,40.);
  mg2->GetYaxis()->SetRangeUser(0,upperRange);
  mg2->GetYaxis()->SetLimits(0,upperRange);
   

    
  TLine *l2 = new TLine();
  l2->SetLineStyle(kDashed);
  l2->SetLineColor(kBlack);

  c2->cd();
  mg2->Draw("APE");
  c2->BuildLegend(0.19,0.71,0.52,0.93)->SetBorderSize(0);
 
  for (int i = 0; i < 3; ++i) {
    l2->DrawLine(25 + i*5, mg2->GetYaxis()->GetXmin(),25+i*5, mg2->GetYaxis()->GetXmax());
  }

  c2->SetFillColor(kWhite);

  //  c2->SaveAs(Form("%s/NeutronDepCosResolution_FullRes_k2.png",output_dir));
 




    TCanvas *c3 = new TCanvas("c3","c3",750, 600);
  TMultiGraph *mg3 = new TMultiGraph();

  TGraphErrors *tge_k3pt0 = new TGraphErrors(n, x, res_k3pt0, ex, err_k3pt0);
  TGraphErrors *tge_k3pt1 = new TGraphErrors(n, x, res_k3pt1, ex, err_k3pt1);
  TGraphErrors *tge_k3pt2 = new TGraphErrors(n, x, res_k3pt2, ex, err_k3pt2);
  TGraphErrors *tge_k3pt3 = new TGraphErrors(n, x, res_k3pt3, ex, err_k3pt3);
  TGraphErrors *tge_k3pt4 = new TGraphErrors(n, x, res_k3pt4, ex, err_k3pt4);
  

  tge_k3pt0->SetTitle(Form("%d #leq p_{T}^{spec}< %d",5,15));              
  tge_k3pt1->SetTitle(Form("%d #leq p_{T}^{spec}< %d",15,25));              
  tge_k3pt2->SetTitle(Form("%d #leq p_{T}^{spec}< %d",25,35));              
  tge_k3pt3->SetTitle(Form("%d #leq p_{T}^{spec}< %d",35,45));              
  tge_k3pt4->SetTitle(Form("%d #leq p_{T}^{spec}",45));              

  tge_k3pt0->SetMarkerColor(kBlue);
  tge_k3pt1->SetMarkerColor(kCyan+1);
  tge_k3pt2->SetMarkerColor(kSpring);
  tge_k3pt3->SetMarkerColor(kOrange);
  tge_k3pt4->SetMarkerColor(kRed);
  tge_k3pt0->SetLineColor(kBlue);
  tge_k3pt1->SetLineColor(kCyan+1);
  tge_k3pt2->SetLineColor(kSpring);
  tge_k3pt3->SetLineColor(kOrange);
  tge_k3pt4->SetLineColor(kRed);
  tge_k3pt0->SetMarkerStyle(21);
  tge_k3pt1->SetMarkerStyle(21);
  tge_k3pt2->SetMarkerStyle(21);
  tge_k3pt3->SetMarkerStyle(21);
  tge_k3pt4->SetMarkerStyle(21);
  
  mg3->Add(tge_k3pt0);
  mg3->Add(tge_k3pt1);
  mg3->Add(tge_k3pt2);
  mg3->Add(tge_k3pt3);
  mg3->Add(tge_k3pt4);
  
  mg3->SetTitle(";N_{neutrons};Res(3#Psi_{1})");

  mg3->GetXaxis()->SetLimits(20.,40.);
  mg3->GetYaxis()->SetRangeUser(0,upperRange);
  mg3->GetYaxis()->SetLimits(0,upperRange);
   

    
  TLine *l3 = new TLine();
  l3->SetLineStyle(kDashed);
  l3->SetLineColor(kBlack);

  c3->cd();
  mg3->Draw("APE");
  c3->BuildLegend(0.19,0.71,0.52,0.93)->SetBorderSize(0);
 
  for (int i = 0; i < 3; ++i) {
    l3->DrawLine(25 + i*5, mg3->GetYaxis()->GetXmin(),25+i*5, mg3->GetYaxis()->GetXmax());
  }

  c3->SetFillColor(kWhite);

  //  c3->SaveAs(Form("%s/NeutronDepCosResolution_FullRes_k3.png",output_dir));
 


}
