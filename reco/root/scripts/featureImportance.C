void featureImportance() {
  
  Double_t gain_x[16] = {0.033398, 0.007163, 0.007608, 0.038799, 0.385574, 0.008194, 0.008007, 0.360377, 0.044438, 0.012516, 0.012859, 0.050624, 0.010157, 0.005183, 0.004929, 0.010176};
  Double_t gain_y[16] = {0.011561, 0.144371, 0.200511, 0.013457, 0.008287, 0.026362, 0.025789, 0.008573, 0.038192, 0.248554, 0.170041, 0.038027, 0.013721, 0.019645, 0.019009, 0.013900};

  Double_t tile_size = 1.14;
  Double_t pos_range[5] = {(-2)*tile_size,(-1)*tile_size,(0.)*tile_size, (1)*tile_size, (2)*tile_size};
  
  TH2F *h_gain_x = new TH2F("h_gain_x","h_gain_x",4, -2*tile_size, 2*tile_size, 4, -2*tile_size, 2*tile_size);
  TH2F *h_gain_y = new TH2F("h_gain_y","h_gain_y",4, -2*tile_size, 2*tile_size, 4, -2*tile_size, 2*tile_size);

  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      int chan = (y*4) + x;
      h_gain_x->SetBinContent(x+1, 4-y, gain_x[chan]);
      h_gain_y->SetBinContent(x+1, 4-y, gain_y[chan]);
    }
  }
  int palette = kLightTemperature;
  gStyle->SetPadRightMargin(0.20);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(palette);
  TCanvas *c1 = new TCanvas("c1","c1",650,500);

  h_gain_x->SetTitle(";x [cm];y [cm]");
  h_gain_x->GetZaxis()->SetTitle("Channel Importance (Gain): x Res");
  h_gain_x->GetZaxis()->SetTitleOffset(1.35);
  h_gain_x->SetContour(99);
  h_gain_x->Draw("colz");
  
  c1->SaveAs(Form("/mnt/c/Users/mwp89/Desktop/ZDC/RPD/FeatureImportance/ChannelImportance_gain_x_palette%d.png",palette));
  delete c1;

  gStyle->SetPadRightMargin(0.2);
  gStyle->SetOptStat(0);
  c1 = new TCanvas("c1","c1",650,500);
  h_gain_y->SetTitle(";x [cm];y [cm]");
  h_gain_y->GetZaxis()->SetTitle("Channel Importance (Gain): y Res");
  h_gain_y->GetZaxis()->SetTitleOffset(1.35);
  h_gain_y->SetContour(99);
  h_gain_y->Draw("colz");

  c1->SaveAs(Form("/mnt/c/Users/mwp89/Desktop/ZDC/RPD/FeatureImportance/ChannelImportance_gain_y_palette%d.png",palette));
  delete c1;

}
