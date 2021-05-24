//RPD resolution plot maker
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "TF1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TPaveStats.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TRandom3.h"
#include "TMultiGraph.h"
#include "TLorentzVector.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TLine.h"

using namespace std;
vector < double > neutron_bins;
vector < double > charge_share_bins;

void DrawChargeMap(TH2D* h1, string label, string out_file);
void DrawPlot(TH1D* h1, bool logy, bool fit, string x_title, string y_axis, string label, string out_file);
void Draw2DPlot(TH2D* h2, string draw_options, bool logz, string x_title, string y_axis, string label, string out_file);
int FindNeutronBin( double n_neutrons );
int FindChargeShareBin( vector<double> rpdSignal);
double makeAngleDiff( double a, double b);
void GaussianFit( TH1D* h1){
  TF1* f1 = new TF1("f1","gaus",-TMath::Pi(),TMath::Pi());
  //h1->Fit("f1","LR");
  //f1->SetRange(f1->GetParameter(1)-1.25*f1->GetParameter(2), f1->GetParameter(1)+1.25*f1->GetParameter(2));
  f1->SetRange(h1->GetMean()-h1->GetRMS(), h1->GetMean()+h1->GetRMS());
  h1->Fit("f1","LR");
}

void GaussianFitGet( TH1D* h1, int idx, double *par, double *par_err ){
  TF1* f1 = new TF1("f1","gaus",-TMath::Pi(),TMath::Pi());
  //h1->Fit("f1","LR");
  //f1->SetRange(f1->GetParameter(1)-1.25*f1->GetParameter(2), f1->GetParameter(1)+1.25*f1->GetParameter(2));
  f1->SetRange(h1->GetMean()-h1->GetRMS(), h1->GetMean()+h1->GetRMS());
  h1->Fit("f1","LR");
  *par = f1->GetParameter(idx);
  *par_err = f1->GetParError(idx);
  return;

}
TStyle* AtlasStyle();
void LoadNeutronBins(){
  neutron_bins.push_back(20); neutron_bins.push_back(25); neutron_bins.push_back(30);
  neutron_bins.push_back(35); neutron_bins.push_back(41);
}
void LoadChargeShareBins(){
  charge_share_bins.push_back(0.); charge_share_bins.push_back(0.2); charge_share_bins.push_back(0.3); charge_share_bins.push_back(0.4);
  charge_share_bins.push_back(0.5); charge_share_bins.push_back(0.6); charge_share_bins.push_back(0.7); charge_share_bins.push_back(1.);
}

TGraphErrors* PlotChargeShareDependence(vector < TH1D* > h1, string yAxis, string outLabel, int parameter = 2);
void PlotNeutronDependence(vector < TH1D* > h1, string yAxis, string outLabel, int parameter = 2);
TGraphErrors* PlotNeutronDependence2(vector < TH1D* > h1, string yAxis, string outLabel, int parameter = 2);
void PlotNeutronDependence_MultiGraph(TGraphErrors *tge[5], string yAxis, string outLabel, double pt_nuc_intervals[5], double yMax);
void PlotNeutronDependenceMean(vector < TH1D* > h1, string yAxis, string outLabel);
TGraphErrors* PlotNeutronDependenceMean2(vector < TH1D* > h1, string yAxis, string outLabel, int ptKick);
void StackNeutronBins( vector < TH1D* > h1, string label, string outputName , bool fit = true);
void StackNeutronBinsPtKick( vector < TH1D* > h1, string label, string outputName, int ptKick , bool fit = true);
void FindCOM(vector <double> rpdSignals,  double &comX, double &comY);
double GetCOMReactionPlane(double comX, double comY, double centerX, double centerY);
double CalcSqrtAverage(double arm_a, double arm_c){
  double avg = (1./(arm_a-arm_c)) * (TMath::Sin(arm_a)-TMath::Sin(arm_c));
  avg = sqrt(avg);
  return avg;
}



void RPD_Plots(){

  char scenario[] = "model_21_maeLoss";

  //RPD file
  TFile* rpd_file = TFile::Open(Form("Output/trees/%s.root",scenario) );
  std::cout << " rpd_file: " << rpd_file << std::endl;
  //Variables to read-out the tree
  TTree* arm_A = (TTree*)rpd_file->Get("ARM A");
  TTree* arm_B = (TTree*)rpd_file->Get("ARM B");
  TTree* avg = (TTree*)rpd_file->Get("Avg RP");
  std::cout << " rpd_file: " << rpd_file << " arm_A " << arm_A << std::endl;
  vector < string > branches;
  branches.push_back("n_incident_neutron");   branches.push_back("X_gen");    branches.push_back("Y_gen");
  branches.push_back("Qx_rec");               branches.push_back("Qy_rec");   branches.push_back("dX");
  branches.push_back("dY");                   branches.push_back("psi_true"); branches.push_back("psi_rec");
  branches.push_back("psi_gen");              branches.push_back("R_rec");    branches.push_back("Pt_nuc");
  branches.push_back("ch_0");                 branches.push_back("ch_1");     branches.push_back("ch_2");
  branches.push_back("ch_3");                 branches.push_back("ch_4");     branches.push_back("ch_5");
  branches.push_back("ch_6");                 branches.push_back("ch_7");     branches.push_back("ch_8");
  branches.push_back("ch_9");                 branches.push_back("ch_10");    branches.push_back("ch_11");
  branches.push_back("ch_12");                branches.push_back("ch_13");    branches.push_back("ch_14");
  branches.push_back("ch_15");                branches.push_back("X_gen_unit");    branches.push_back("Y_gen_unit");
  branches.push_back("ch_0_raw");                 branches.push_back("ch_1_raw");     branches.push_back("ch_2_raw");
  branches.push_back("ch_3_raw");                 branches.push_back("ch_4_raw");     branches.push_back("ch_5_raw");
  branches.push_back("ch_6_raw");                 branches.push_back("ch_7_raw");     branches.push_back("ch_8_raw");
  branches.push_back("ch_9_raw");                 branches.push_back("ch_10_raw");    branches.push_back("ch_11_raw");
  branches.push_back("ch_12_raw");                branches.push_back("ch_13_raw");    branches.push_back("ch_14_raw");
  branches.push_back("ch_15_raw");                

  std::map < string, double > map_A;
  std::map < string, double > map_B;
  std::map < string, double > map_M;

  LoadNeutronBins();
  LoadChargeShareBins();

  vector < TH1D* >  resoRP_gen_chargeShare;
  
  vector < TH1D* > resoRP_true_gen;
  vector < TH1D* > resoRP_true_A;
  vector < TH1D* > resoRP_true_B;
  vector < TH1D* > resoRP_gen_A;
  vector < TH1D* > resoRP_gen_B;
  vector < TH1D* > resoRP_averageTruth;
  vector < TH1D* > resoRP_averageGen;
  vector < TH1D* > resoRP_averageTruth_ptKick[6];
  vector < TH1D* > resoRP_averageGen_ptKick[6];
  vector < TH1D* > resoRP_A_Gen_ptKick[6];
  vector < TH1D* > resoRP_A_Truth_ptKick[6];

  vector < TH1D* > resoRP_COM_A_Gen_ptKick[6];
  
  vector < TH1D* > resoRP_averageArms;
  vector < TH1D* > resoRP_averageArms_ptKick[6];

  vector < TH1D* > resoX_gen_A;   vector < TH1D* > resoY_gen_A;
  vector < TH1D* > resoX_gen_B;   vector < TH1D* > resoY_gen_B;
  vector < TH1D* > resoRadius_gen_A;   vector < TH1D* > resoRadius_gen_B;

  vector < TH2D* > resoRP_true_vs_RP_true;
  vector < TH2D* > resoRP_gen_vs_RP_gen;
  vector < TH2D* > resoRP_gen_vs_RP_true;

  vector < TH2D* > QxA_vs_QxC;
  vector < TH2D* > QyA_vs_QyC;
  vector < TH2D* > gen_QxA_vs_QxC;
  vector < TH2D* > gen_QyA_vs_QyC;


  TH2D* rpd_signal_A;
  TH2D* rpd_signal_B;
  TH2D* rpd_signal_A_raw;
  TH2D* rpd_signal_B_raw;
  TH2D* rpd_signal_A_allEvents = new TH2D("rpdA_allEvents","",4,-2,2,4,-2,2);
  TH2D* rpd_signal_B_allEvents = new TH2D("rpdC_allEvents","",4,-2,2,4,-2,2);
  TH2D* rpd_gen_pos_unit_A_allEvents = new TH2D("rpdA_gen_pos_unitVec_allEvents","",10,-10,10,10,-10,10);
  TH2D* rpd_gen_pos_unit_B_allEvents = new TH2D("rpdC_gen_pos_unitVec_allEvents","",10,-10,10,10,-10,10);
  TH2D* rpd_gen_pos_A_allEvents = new TH2D("rpdA_gen_pos_allEvents","",16,-1,1,16,-1,1);
  TH2D* rpd_gen_pos_B_allEvents = new TH2D("rpdC_gen_pos_allEvents","",4,-2,2,4,-2,2);

  string histoname; string histo_title; int reso_binning = 100;
  //Unbinned histograms

  for (int i = 0; i < 7; ++i) {
    //Reaction Plane Gen - Rec   
    char histoname[256];
    sprintf(histoname, "ResolutionGen_RP_chargeShareBin%d",i);
    string histo_title = "#psi_{0}^{A-Gen} - #psi_{0}^{A-Rec}";
    resoRP_gen_chargeShare.push_back(new TH1D(histoname, histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
  }
  
  //Binned histograms
  for(int i = 0; i < (int)neutron_bins.size()-1; i++){
    ostringstream nl, nu;
    nl << neutron_bins.at(i);
    nu << neutron_bins.at(i+1);
    //Reaction Plane reso - a la STAR -- differentiated by pt kick
    for (int ptKick = 0; ptKick < 6; ++ptKick) {
      histoname = "ResoSTAR_RP_ACs_n_in_" + nl.str() + "_" + nu.str() + "_ptKick" + std::to_string(ptKick);
      histo_title = "<cos (#psi_{0}^{Rec-A} - #psi_{0}^{Rec-C})>";
      resoRP_averageArms_ptKick[ptKick].push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -1, 1.));

      histoname = "ResolutionTrue_RP_Avg_n_in_" + nl.str() + "_" + nu.str() + "_ptKick" + std::to_string(ptKick);
      histo_title = "#psi_{0}^{Truth} - #psi_{0}^{Rec-AC}";
      resoRP_averageTruth_ptKick[ptKick].push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
      histoname = "ResolutionGen_RP_Avg_in_" + nl.str() + "_" + nu.str() + "_ptKick" + std::to_string(ptKick);
      histo_title = "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC}";
      resoRP_averageGen_ptKick[ptKick].push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));

      histoname = "ResolutionGen_RP_A_in_" + nl.str() + "_" + nu.str() + "_ptKick" + std::to_string(ptKick);
      histo_title = "#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}";
      resoRP_A_Gen_ptKick[ptKick].push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));

      histoname = "ResolutionTruth_RP_A_in_" + nl.str() + "_" + nu.str() + "_ptKick" + std::to_string(ptKick);
      histo_title = "#psi_{0}^{Truth-A} - #psi_{0}^{Rec-A}";
      resoRP_A_Truth_ptKick[ptKick].push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
    
      histoname = "ResolutionGen_RP_COM_A_in_" + nl.str() + "_" + nu.str() + "_ptKick" + std::to_string(ptKick);
      histo_title = "#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}";
      resoRP_COM_A_Gen_ptKick[ptKick].push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));

      
    }
    
    //Reaction Plane reso - a la STAR
    histoname = "ResoSTAR_RP_ACs_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "<cos (#psi_{0}^{Rec-A} - #psi_{0}^{Rec-C})>";
    resoRP_averageArms.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -1, 1.));
    //Reaction Plane Truth - Gen
    histoname = "TruthvsGen_RP_A_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{Truth} - #psi_{0}^{A-Gen}";
    resoRP_true_gen.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
    //Reaction Plane Truth - Rec
    histoname = "ResolutionTruth_RP_A_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{A-Truth} - #psi_{0}^{A-Rec}";
    resoRP_true_A.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
    histoname = "ResolutionTruth_RP_B_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{C-Truth} - #psi_{0}^{C-Rec}";
    resoRP_true_B.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
    //Reaction Plane Gen - Rec
    histoname = "ResolutionGen_RP_A_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{A-Gen} - #psi_{0}^{A-Rec}";
    resoRP_gen_A.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
    histoname = "ResolutionGen_RP_B_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{C-Truth} - #psi_{0}^{C-Rec}";
    resoRP_gen_B.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
    //Reaction Plane Truth - Rec
    histoname = "ResolutionTrue_RP_Avg_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{Truth} - #psi_{0}^{Rec-AC}";
    resoRP_averageTruth.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));
    histoname = "ResolutionGen_RP_Avg_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC}";
    resoRP_averageGen.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -TMath::Pi(), TMath::Pi()));

    //X axis
    histoname = "ResolutionGen_X_A_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "Q_{x}^{A-Gen} - Q_{x}^{A-Rec}";
    resoX_gen_A.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -2.5, 2.5));
    histoname = "ResolutionGen_X_B_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "Q_{x}^{C-Gen} - Q_{x}^{C-Rec}";
    resoX_gen_B.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -2.5, 2.5));
    //Y axis
    histoname = "ResolutionGen_Y_A_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "Q_{y}^{A-Gen} - Q_{y}^{A-Rec}";
    resoY_gen_A.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -2.5, 2.5));
    histoname = "ResolutionGen_Y_B_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "Q_{y}^{C-Gen} - Q_{y}^{C-Rec}";
    resoY_gen_B.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -2.5, 2.5));
    //R
    histoname = "ResolutionGen_R_A_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "|Q|^{A-Gen} - |Q|^{A-Rec}";
    resoRadius_gen_A.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -5., 5.));
    histoname = "ResolutionGen_R_C_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "|Q|^{C-Gen} - |Q|^{C-Rec}";
    resoRadius_gen_B.push_back(new TH1D(histoname.c_str(), histo_title.c_str(), reso_binning, -5., 5.));

    // 2D Histograms
    // resoRP_true_vs_RP_true;
    histoname = "ResolutionTruth_vs_Truth_RP_AC_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{Truth} - #psi_{0}^{Rec-AC} vs #psi_{0}^{Truth}";
    resoRP_true_vs_RP_true.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),32,-TMath::Pi(),TMath::Pi(),32,-TMath::Pi(),TMath::Pi()));
    // resoRP_gen_vs_RP_gen;
    histoname = "ResolutionGen_vs_Gen_RP_AC_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC} vs #psi_{0}^{Gen-AC}";
    resoRP_gen_vs_RP_gen.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),32,-TMath::Pi(),TMath::Pi(),32,-TMath::Pi(),TMath::Pi()));
    // resoRP_gen_vs_RP_true;
    histoname = "ResolutionGen_vs_Truth_RP_AC_n_in_" + nl.str() + "_" + nu.str();
    histo_title = "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC} vs #psi_{0}^{Truth}";
    resoRP_gen_vs_RP_true.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),32,-TMath::Pi(),TMath::Pi(),32,-TMath::Pi(),TMath::Pi()));
    // QxA vs QxB
    histoname = "QxA_vs_QxC_n_in" + nl.str() + "_" + nu.str();
    histo_title = "Q_{x}^{A} vs Q_{x}^{C}";
    //    QxA_vs_QxC.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),60,-6,6,60,-6,6));
    QxA_vs_QxC.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),10,-10,10,10,-10,10));
    //QyA vs QyB
    histoname = "QyA_vs_QyC_n_in" + nl.str() + "_" + nu.str();
    histo_title = "Q_{y}^{A} vs Q_{y}^{C}";
    QyA_vs_QyC.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),10,-10,10,10,-10,10));
    // Gen QxA vs QxB
    histoname = "Gen_QxA_vs_Gen_QxC_n_in" + nl.str() + "_" + nu.str();
    histo_title = "Q_{x}^{Gen-A} vs Q_{x}^{Gen-C}";
    gen_QxA_vs_QxC.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),10,-10,10,10,-10,10));
    // Gen QyA vs QyB
    histoname = "Gen_QyA_vs_Gen_QyC_n_in" + nl.str() + "_" + nu.str();
    histo_title = "Q_{y}^{Gen-A} vs Q_{y}^{Gen-C}";
    gen_QyA_vs_QyC.push_back(new TH2D(histoname.c_str(), histo_title.c_str(),10,-10,10,10,-10,10));
  }



  //Fitting and plotting
  TStyle* atlasStyle = AtlasStyle();
  gROOT->ForceStyle();
  atlasStyle->cd();

  string extension = ".png";
  string label, pName;
  string scenarioString(scenario);
  string folderPrefix = "/mnt/c/Users/mwp89/Desktop/Thesis/RPD/Plots/"; 
  //  string folderChargeMap =  folderPrefix + scenarioString + "/ChargeMaps/";
  //  string folderChargeMap =  folderPrefix + scenarioString + "/ChargeMaps_4mmOrigin/";
  //  string folderChargeMap =  folderPrefix + scenarioString + "/ChargeMaps_misreconstruction/";
  //  string folderChargeMap =  folderPrefix + scenarioString + "/ChargeMaps_misreconstruction_2arm/";
  string folderChargeMap =  folderPrefix + scenarioString + "/ChargeMaps_misreconstruction_com/";
  string folder1d =  folderPrefix + scenarioString + "/1DPlots/";
  string folder2d = folderPrefix + scenarioString + "/2D_Plots/";
  string folderMainRes = folderPrefix + scenarioString + "/MainResolution/";
  string folderNeutronDep = folderPrefix + scenarioString + "/NeutronDependence/";
  string folderChargeShareDep = folderPrefix + scenarioString + "/ChargeShareDependence/";
  
  // Tree Branch Assignment
  for(int i = 0; i < (int)branches.size(); i++){
    std::cout << " branches.size() " << branches.size() << " address " << branches.at(i).c_str() << std::endl;
    arm_A->SetBranchAddress(branches.at(i).c_str(), &map_A[branches.at(i).c_str()]);
    arm_B->SetBranchAddress(branches.at(i).c_str(), &map_B[branches.at(i).c_str()]);
  }
  int nBinChargeShareA, nBinChargeShareB;
  int nBinA, nBinB, nBinMean;
  double R_genA, R_recA, dXA, dYA;
  double R_genB, R_recB, dXB, dYB;
  double tdiffAzA, tdiffAzB, tdiffAzAB;
  double gdiffAzA, gdiffAzB, gdiffAzAB;
  double gdiffAzA_com, gdiffAzB_com, gdiffAzAB_com;
  double diffTGA; double pt_nucA; double pt_nucB;
  vector<double> rpd_A;
  vector<double> rpd_B;
  vector<double> rpd_A_raw;
  vector<double> rpd_B_raw;
  std::cout << " TREE ENTRIES: " << (int)arm_A->GetEntries() << std::endl;
  int entries = (int)arm_A->GetEntries();

  double pt_nuc_intervals[5] = {0.005,0.015,0.025,0.035,0.045}; // for samplings of 10, 20, 30, 40 MeV
  int ptNucInterval = -1;

  int picCount = 0;
  for(int i = 0; i < (int)arm_A->GetEntries(); i++){ //i < 100; i++) { //
    if (i % 1000 == 0) std::cout << " event " << i << std::endl;
      arm_A->GetEvent(i);
      arm_B->GetEvent(i);
      //      if(i == 1e6) break;
      if(i == entries) break;
      //cout<< "n_incident_neutron: " << map_A["n_incident_neutron"] << " ; " << map_B["n_incident_neutron"] << endl;

      char histoName[256];
      sprintf(histoName,"rpdSignal_A_event%d",i);
      rpd_signal_A = new TH2D(histoName,histoName,4,-2,2,4,-2,2);
      sprintf(histoName,"rpdSignal_B_event%d",i);
      rpd_signal_B = new TH2D(histoName,histoName,4,-2,2,4,-2,2);
      sprintf(histoName,"rpdSignal_A_raw_event%d",i);
      rpd_signal_A_raw = new TH2D(histoName,histoName,4,-2,2,4,-2,2);
      sprintf(histoName,"rpdSignal_B_raw_event%d",i);
      rpd_signal_B_raw = new TH2D(histoName,histoName,4,-2,2,4,-2,2);
      
      //Compute variables
      nBinA = FindNeutronBin( map_A["n_incident_neutron"]);
      nBinB = FindNeutronBin( map_B["n_incident_neutron"]);
      nBinMean = (nBinA+nBinB)/2;
      float nNeutronsA = map_A["n_incident_neutron"];
      float nNeutronsB = map_B["n_incident_neutron"];
      dXA = map_A["X_gen"] - map_A["Qx_rec"];
      dYA = map_A["Y_gen"] - map_A["Qy_rec"];
      R_genA = sqrt(pow(map_A["X_gen"],2)+pow(map_A["Y_gen"],2));
      R_recA = sqrt(pow(map_A["Qx_rec"],2)+pow(map_A["Qy_rec"],2));
      pt_nucA = map_A["Pt_nuc"];
      pt_nucB = map_B["Pt_nuc"];


      for (int ch = 0; ch < 16; ++ch) {
	char channel[256];
	sprintf(channel,"ch_%d",ch);
	string chan(channel);
	rpd_A.push_back(map_A[chan]);
	rpd_B.push_back(map_B[chan]);
	char channel2[256];
	sprintf(channel2,"ch_%d_raw",ch);
	string chan2(channel2);
	rpd_A_raw.push_back(map_A[chan2]);
	rpd_B_raw.push_back(map_B[chan2]);
      }
      nBinChargeShareA = FindChargeShareBin(rpd_A);
      nBinChargeShareB = FindChargeShareBin(rpd_B);
      //      if (i < 25 ) std::cout << " event " << i << " Qx_rec " << map_A["Qx_rec"] << " Qy_rec " << map_A["Qy_rec"] << " psi_rec " << map_A["psi_rec"]  << std::endl;
      if (map_A["Pt_nuc"] <= pt_nuc_intervals[0]) ptNucInterval = 0;
      else if (map_A["Pt_nuc"] > pt_nuc_intervals[0] && map_A["Pt_nuc"] <= pt_nuc_intervals[1]) ptNucInterval = 1;
      else if (map_A["Pt_nuc"] > pt_nuc_intervals[1] && map_A["Pt_nuc"] <= pt_nuc_intervals[2]) ptNucInterval = 2;
      else if (map_A["Pt_nuc"] > pt_nuc_intervals[2] && map_A["Pt_nuc"] <= pt_nuc_intervals[3]) ptNucInterval = 3;
      else if (map_A["Pt_nuc"] > pt_nuc_intervals[3] && map_A["Pt_nuc"] <= pt_nuc_intervals[4]) ptNucInterval = 4;
      else ptNucInterval = 5;
	    
      dXB = map_B["X_gen"] - map_B["Qx_rec"];
      dYB = map_B["Y_gen"] - map_B["Qy_rec"];
      R_genB = sqrt(pow(map_B["X_gen"],2)+pow(map_B["Y_gen"],2));
      R_recB = sqrt(pow(map_B["Qx_rec"],2)+pow(map_B["Qy_rec"],2));

      TVector3 *rpA_rec = new TVector3( TMath::Cos(map_A["psi_rec"]), TMath::Sin(map_A["psi_rec"]), 0 );
      TVector3 *rpB_rec = new TVector3( -TMath::Cos(map_B["psi_rec"]), -TMath::Sin(map_B["psi_rec"]), 0 );
      TVector3 *rpAvg = new TVector3(0,0,0);
      *rpAvg += *rpA_rec;
      *rpAvg += *rpB_rec;

      TVector3 *rpA_gen = new TVector3( TMath::Cos(map_A["psi_gen"]), TMath::Sin(map_A["psi_gen"]), 0 );
      TVector3 *rpB_gen = new TVector3( -TMath::Cos(map_B["psi_gen"]), -TMath::Sin(map_B["psi_gen"]), 0 );
      TVector3 *rpAvg_gen = new TVector3(0,0,0);
      *rpAvg_gen += *rpA_gen;
      *rpAvg_gen += *rpB_gen;

      double RPflip;
      if (map_B["psi_rec"] > 0) RPflip = map_B["psi_rec"]-TMath::Pi();
      else RPflip = map_B["psi_rec"] + TMath::Pi();
      // RPflip only necessary when comparing truth to reco or gen on the C side. For reco to gen comparisons don't flip
      
      tdiffAzA = makeAngleDiff(map_A["psi_true"], map_A["psi_rec"]);
      //      std::cout << "APsi_true " << map_A["psi_true"] << " APsi_rec " << map_A["psi_rec"] << " diff " << tdiffAzA << std::endl;
      tdiffAzB = makeAngleDiff(map_B["psi_true"], RPflip);

      //            std::cout << "BPsi_true " << map_B["psi_true"] << " BPsi_rec " << RPflip << " diff " << tdiffAzB << std::endl;

      tdiffAzAB = makeAngleDiff(map_A["psi_true"], rpAvg->Phi());

      gdiffAzA = makeAngleDiff(map_A["psi_gen"], map_A["psi_rec"]);
      //      gdiffAzB = makeAngleDiff(map_B["psi_gen"], RPflip);
      gdiffAzB = makeAngleDiff(map_B["psi_gen"], map_B["psi_rec"]);
      gdiffAzAB = makeAngleDiff(rpAvg_gen->Phi(), rpAvg->Phi());
      //      std::cout << "BPsi_gen " << map_B["psi_gen"] << " BPsi_rec " << RPflip << " diff " << gdiffAzB << std::endl;
      diffTGA = makeAngleDiff(map_A["psi_true"],map_A["psi_gen"]);

      double comPhi = 0;
      double comX = 0;       double comY = 0;     
      FindCOM(rpd_A, comX, comY);
      comPhi = GetCOMReactionPlane(comX, comY, 0., -0.5);
      gdiffAzA_com = makeAngleDiff(map_A["psi_gen"], comPhi);
      
      //      std::cout << " event " << i << std::endl;
      for (int ch = 0; ch < 16; ++ch) {
	float x; float y;
	if (ch < 4) y = 1.5;
	else if (ch >= 4 && ch < 8) y = 0.5;
	else if (ch >= 8 && ch < 12) y = -0.5;
	else y = -1.5;
	if (ch % 4 == 0) x = 1.5;
	else if (ch % 4 == 1) x = 0.5;
	else if (ch % 4 == 2) x = -0.5;
	else if (ch % 4 == 3) x = -1.5;
	rpd_signal_A->Fill(x,y,rpd_A[ch]);
	rpd_signal_B->Fill(x,y,rpd_B[ch]);
	rpd_signal_A_raw->Fill(x,y,rpd_A_raw[ch]);
	rpd_signal_B_raw->Fill(x,y,rpd_B_raw[ch]);
	//	if (y == -0.5)	std::cout << " row 3: rawA " << rpd_A_raw[ch] << " subtrA " << rpd_A[ch] << std::endl;
	//	if (y == -1.5)	std::cout << " row 4: rawA " << rpd_A_raw[ch] << " subtrA " << rpd_A[ch] << " calcSignal " << rpd_A_raw[ch] - rpd_A_raw[ch-4] <<  std::endl;

	rpd_signal_A_allEvents->Fill(x,y,rpd_A[ch]);
	rpd_signal_B_allEvents->Fill(x,y,rpd_B[ch]);
      }
      rpd_gen_pos_A_allEvents->Fill(map_A["X_gen"]*1e-1,map_A["Y_gen"]*1e-1);
      rpd_gen_pos_B_allEvents->Fill(map_B["X_gen"]*1e-1,map_B["Y_gen"]*1e-1);
      rpd_gen_pos_unit_A_allEvents->Fill(map_A["X_gen_unit"],map_A["Y_gen_unit"]);
      rpd_gen_pos_unit_B_allEvents->Fill(map_B["X_gen_unit"],map_B["Y_gen_unit"]);
      resoRP_true_gen.at(nBinA)->Fill(diffTGA);
      resoRP_true_A.at(nBinA)->Fill(tdiffAzA);
      resoRP_true_B.at(nBinB)->Fill(tdiffAzB);

      resoRP_gen_chargeShare.at(nBinChargeShareA)->Fill(gdiffAzA);
      resoRP_gen_chargeShare.at(nBinChargeShareB)->Fill(gdiffAzB);
      
      resoRP_gen_A.at(nBinA)->Fill(gdiffAzA);
      resoRP_gen_B.at(nBinB)->Fill(gdiffAzB);
      resoRP_averageTruth.at(nBinMean)->Fill(tdiffAzAB);
      resoRP_averageGen.at(nBinMean)->Fill(gdiffAzAB);

      resoRP_averageTruth_ptKick[ptNucInterval].at(nBinMean)->Fill(tdiffAzAB);
      resoRP_averageGen_ptKick[ptNucInterval].at(nBinMean)->Fill(gdiffAzAB);
      
      resoRP_A_Gen_ptKick[ptNucInterval].at(nBinMean)->Fill(gdiffAzA);
      resoRP_A_Truth_ptKick[ptNucInterval].at(nBinMean)->Fill(tdiffAzA);

      resoRP_COM_A_Gen_ptKick[ptNucInterval].at(nBinMean)->Fill(gdiffAzA_com);
      
      resoX_gen_A.at(nBinA)->Fill(dXA);
      resoX_gen_B.at(nBinB)->Fill(dXB);
      resoY_gen_A.at(nBinA)->Fill(dYA);
      resoY_gen_B.at(nBinB)->Fill(dYB);

      resoRadius_gen_A.at(nBinA)->Fill(R_genA-R_recA);
      resoRadius_gen_B.at(nBinB)->Fill(R_genB-R_recB);


      //2D plots
      resoRP_true_vs_RP_true.at(nBinMean)->Fill(map_A["psi_true"], tdiffAzAB);
      resoRP_gen_vs_RP_gen.at(nBinMean)->Fill(rpAvg_gen->Phi(), gdiffAzAB);
      resoRP_gen_vs_RP_true.at(nBinMean)->Fill(map_A["psi_true"], gdiffAzAB);
      
      //      double RPflip_orig = map_B["psi_rec"]-TMath::Pi();


      //      if(RPflip_orig < -TMath::Pi()) RPflip_orig += TMath::TwoPi();
      //      if(RPflip_orig > TMath::Pi()) RPflip_orig -= TMath::TwoPi();
      resoRP_averageArms.at(nBinMean)->Fill(TMath::Cos(map_A["psi_rec"]-RPflip));
      //      std::cout << " A_psi " << map_A["psi_rec"] << " B_psi " << map_B["psi_rec"] << " RPflipB " << RPflip << " RBflipB_orig " << RPflip_orig << " subtracted " << TMath::Cos(map_A["psi_rec"]-RPflip) << std::endl;;

      resoRP_averageArms_ptKick[ptNucInterval].at(nBinMean)->Fill(TMath::Cos(map_A["psi_rec"]-RPflip));


      QxA_vs_QxC.at(nBinMean)->Fill(map_A["Qx_rec"], map_B["Qx_rec"]);
      QyA_vs_QyC.at(nBinMean)->Fill(map_A["Qy_rec"], map_B["Qy_rec"]);
      gen_QxA_vs_QxC.at(nBinMean)->Fill(map_A["X_gen"], map_B["X_gen"]);
      gen_QyA_vs_QyC.at(nBinMean)->Fill(map_A["Y_gen"], map_B["Y_gen"]);
      std::stringstream psi_true_A;
      psi_true_A << std::fixed << std::setprecision(2) <<  map_A["psi_true"];
      double RPflip_true;
      if (map_B["psi_true"] > 0) RPflip_true = map_B["psi_true"]-TMath::Pi();
      else RPflip_true = map_B["psi_true"] + TMath::Pi();

      std::stringstream psi_true_B;
      psi_true_B << std::fixed << std::setprecision(2) <<  RPflip_true;
      std::stringstream psi_gen_A;
      psi_gen_A << std::fixed << std::setprecision(2) <<  map_A["psi_gen"];
      std::stringstream psi_gen_B;
      psi_gen_B << std::fixed << std::setprecision(2) <<  map_B["psi_gen"];
      std::stringstream psi_rec_A;
      psi_rec_A << std::fixed << std::setprecision(2) <<  map_A["psi_rec"];
      std::stringstream psi_rec_B;
      psi_rec_B << std::fixed << std::setprecision(2) <<  map_B["psi_rec"];
      std::stringstream x_pos_A;
      x_pos_A << std::fixed << std::setprecision(2) <<  map_A["X_gen"] * 1e-1;
      std::stringstream y_pos_A;
      y_pos_A << std::fixed << std::setprecision(2) <<  map_A["Y_gen"] * 1e-1;
      std::stringstream x_pos_B;
      x_pos_B << std::fixed << std::setprecision(2) <<  map_B["X_gen"] * 1e-1;
      std::stringstream y_pos_B;
      y_pos_B << std::fixed << std::setprecision(2) <<  map_B["Y_gen"] * 1e-1;
      std::stringstream res_A;
      res_A << std::fixed << std::setprecision(2) <<  gdiffAzA;
      std::stringstream res_C;
      res_C << std::fixed << std::setprecision(2) <<  gdiffAzB;
      std::stringstream x_pos_COM;
      x_pos_COM << std::fixed << std::setprecision(2) <<  comX;
      std::stringstream y_pos_COM;
      y_pos_COM << std::fixed << std::setprecision(2) <<  comY;
      std::stringstream res_A_COM;
      res_A_COM << std::fixed << std::setprecision(2) <<  gdiffAzA_com;
      std::stringstream phi_COM;
      phi_COM << std::fixed << std::setprecision(2) <<  comPhi;
      
      int pt_nuc_A  = (int) (map_A["Pt_nuc"] * 1e3);
      int pt_nuc_B  = (int) (map_B["Pt_nuc"] * 1e3);
      int n_neutrons_A  = (int) map_A["n_incident_neutron"];
      int n_neutrons_B  = (int) map_B["n_incident_neutron"];

      // only save the first 15
      if (picCount < 50) {
	//	if (TMath::Abs(map_A["X_gen"]) > 5 || TMath::Abs(map_A["Y_gen"]) > 5 ) {

	if (TMath::Abs(gdiffAzA_com) > 2.5 ) {
	  //	  	std::cout << " event " << i << " residA " << gdiffAzA <<  " residB " << gdiffAzB << " residAC " << gdiffAzAB << " CosA_psi_gen " << TMath::Cos(map_A["psi_gen"]) << " SinA_psi_gen " << TMath::Sin(map_A["psi_gen"]) <<  " -CosB_psi_gen " << -TMath::Cos(map_B["psi_gen"]) <<  " -SinB_psi_gen " << -TMath::Sin(map_B["psi_gen"]) << " rpAvg_gen_x: " << rpAvg_gen->X() << " rpAvg_gen_y: " << rpAvg_gen->Y() << " CosA_psi_rec " << TMath::Cos(map_A["psi_rec"]) << " SinA_psi_rec " << TMath::Sin(map_A["psi_rec"]) <<  " -CosB_psi_rec " << -TMath::Cos(map_B["psi_rec"]) <<  " -SinB_psi_rec " << -TMath::Sin(map_B["psi_rec"]) << " rpAvg_rec_x: " << rpAvg->X() << " rpAvg_y: " << rpAvg->Y() << std::endl;
	  //  std::cout << " event " << i << " residA " << gdiffAzA <<  " residB " << gdiffAzB << " residAC " << gdiffAzAB << " A_psi_gen " << map_A["psi_gen"] << " B_psi_gen " << map_B["psi_gen"] <<  " A_psi_rec " << map_A["psi_rec"] <<  " A_psi_rec " << map_B["psi_rec"] << " CosA_psi_gen " << TMath::Cos(map_A["psi_gen"]) << " SinB_psi_gen " << TMath::Sin(map_B["psi_gen"]) <<  " -CosA_psi_rec " << -TMath::Cos(map_A["psi_rec"]) <<  " -SinB_psi_rec " << -TMath::Sin(map_B["psi_rec"]) << std::endl;
		//	  label = "#splitline{#splitline{Incident x: " + x_pos_A.str() + " cm; Incident y: " + y_pos_A.str() + " cm}{#psi^{Truth}: " + psi_true_A.str() + " rad; #psi^{Gen-A}: " + psi_gen_A.str() + " rad; #psi^{Rec-A}: " + psi_rec_A.str() + " rad}}{#splitline{# of Neutrons: " + to_string(n_neutrons_A) + "; #it{p}_{T}^{Nuc}: " + to_string(pt_nuc_A) + " MeV}{#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}: " + res_A.str() + " rad}}";
	  //	  label = "#splitline{#splitline{Incident x: " + x_pos_A.str() + " cm; Incident y: " + y_pos_A.str() + " cm}{#psi^{Truth}: " + psi_true_A.str() + " rad; #psi^{Gen-A}: " + psi_gen_A.str() + " rad; #psi^{Rec-A}: " + psi_rec_A.str() + " rad}}{#splitline{COM x: " + x_pos_COM.str() + " cm; COM y: " + y_pos_COM.str() + " #psi^{Rec-A-COM}: " + to_string(comPhi) + " ; }{#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}: " + res_A.str() + " rad}}";
	  label = "#splitline{#splitline{# of Neutrons: " + to_string(n_neutrons_A) + "; #it{p}_{T}^{Nuc}: " + to_string(pt_nuc_A) + " MeV}{#psi^{Truth}: " + psi_true_A.str() + " rad; #psi^{Gen-A}: " + psi_gen_A.str() + " rad}}{#splitline{#psi^{Rec-A}: " + psi_rec_A.str() + " rad; #psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}: " + res_A.str() + " rad}{#psi^{Rec-A-COM}: " + phi_COM.str() + " rad; #psi_{0}^{Gen-A} - #psi_{0}^{Rec-A-COM}: " + res_A_COM.str() + " rad}}";
	  //	label = "##splitline{Incident x: " + x_pos_A.str() + " Incident y: " + y_pos_A.str() + "}{#psi^{Truth}: " + psi_true_A.str() + " rad; #psi^{Gen-A}: " + psi_gen_A.str() + " rad; #psi^{Rec-A}: " + psi_rec_A.str() + "}";
	  pName = "ChargeMapA_event_" + to_string(i) + "_neutrons_" + to_string(n_neutrons_A) + "_ptNuc_" + to_string(pt_nuc_A) + "_resA_" + res_A.str() + extension;
	  DrawChargeMap(rpd_signal_A, label, folderChargeMap + pName);
	  label = "#splitline{#splitline{Incident x: " + x_pos_B.str() + " cm; Incident y: " + y_pos_B.str() + " cm}{#psi^{Truth}: " + psi_true_B.str() + " rad; #psi^{Gen-C}: " + psi_gen_B.str() + " rad; #psi^{Rec-C}: " + psi_rec_B.str() + " rad}}{#splitline{# of Neutrons: " + to_string(n_neutrons_B) + "; #it{p}_{T}^{Nuc}: " + to_string(pt_nuc_B) + " MeV}{#psi_{0}^{Gen-C} - #psi_{0}^{Rec-C}: " + res_C.str() + " rad}}";
	  //	label = "#splitline{Incident x: " + x_pos_B.str() + " Incident y: " + y_pos_B.str() + "}#psi^{Truth}: " + psi_true_B.str() + " rad; #psi^{Gen-A}: " + psi_gen_B.str() + " rad; #psi^{Rec-A}: " + psi_rec_B.str() + "}";
	  pName = "ChargeMapC_event_" + to_string(i) + "_neutrons_" + to_string(n_neutrons_A) + "_ptNuc_" + to_string(pt_nuc_A) + "_resC_" + res_C.str() + extension;
	  DrawChargeMap(rpd_signal_B, label, folderChargeMap + pName);
	  picCount++;
	}/*
	if (TMath::Abs(gdiffAzB) > 2.5 ) {
	//	if (TMath::Abs(map_B["X_gen"]) > 5 || TMath::Abs(map_B["Y_gen"]) > 5 ) {
	  label = "#splitline{#splitline{Incident x: " + x_pos_B.str() + " cm; Incident y: " + y_pos_B.str() + " cm}{#psi^{Truth}: " + psi_true_B.str() + " rad; #psi^{Gen-C}: " + psi_gen_B.str() + " rad; #psi^{Rec-C}: " + psi_rec_B.str() + " rad}}{#splitline{# of Neutrons: " + to_string(n_neutrons_B) + "; #it{p}_{T}^{Nuc}: " + to_string(pt_nuc_B) + " MeV}{#psi_{0}^{Gen-C} - #psi_{0}^{Rec-C}: " + res_C.str() + " rad}}";
	  //	label = "#splitline{Incident x: " + x_pos_B.str() + " Incident y: " + y_pos_B.str() + "}#psi^{Truth}: " + psi_true_B.str() + " rad; #psi^{Gen-A}: " + psi_gen_B.str() + " rad; #psi^{Rec-A}: " + psi_rec_B.str() + "}";
	  pName = "ChargeMapC_event_" + to_string(i) + "_neutrons_" + to_string(n_neutrons_A) + "_ptNuc_" + to_string(pt_nuc_A) + "_resC_" + res_C.str() + extension;
	  DrawChargeMap(rpd_signal_B, label, folderChargeMap + pName);
	  picCount++;
	}
	 */
      }
      if (i < 0) {
	label = "#splitline{#splitline{Incident x: " + x_pos_A.str() + " cm; Incident y: " + y_pos_A.str() + " cm}{#psi^{Truth}: " + psi_true_A.str() + " rad; #psi^{Gen-A}: " + psi_gen_A.str() + " rad; #psi^{Rec-A}: " + psi_rec_A.str() + " rad}}{#splitline{# of Neutrons: " + to_string(n_neutrons_A) + "; #it{p}_{T}^{Nuc}: " + to_string(pt_nuc_A) + " MeV}{}}";
	//	label = "##splitline{Incident x: " + x_pos_A.str() + " Incident y: " + y_pos_A.str() + "}{#psi^{Truth}: " + psi_true_A.str() + " rad; #psi^{Gen-A}: " + psi_gen_A.str() + " rad; #psi^{Rec-A}: " + psi_rec_A.str() + "}";
	pName = "ChargeMapA_Raw_event_" + to_string(i) + "_neutrons_" + to_string(n_neutrons_A) + "_ptNuc_" + to_string(pt_nuc_A) + extension;
	DrawChargeMap(rpd_signal_A_raw, label, folderChargeMap + pName);
	
	label = "#splitline{#splitline{Incident x: " + x_pos_B.str() + " cm; Incident y: " + y_pos_B.str() + " cm}{#psi^{Truth}: " + psi_true_B.str() + " rad; #psi^{Gen-A}: " + psi_gen_B.str() + " rad; #psi^{Rec-A}: " + psi_rec_B.str() + " rad}}{#splitline{# of Neutrons: " + to_string(n_neutrons_B) + "; #it{p}_{T}^{Nuc}: " + to_string(pt_nuc_B) + " MeV}{}}";
	//	label = "#splitline{Incident x: " + x_pos_B.str() + " Incident y: " + y_pos_B.str() + "}#psi^{Truth}: " + psi_true_B.str() + " rad; #psi^{Gen-A}: " + psi_gen_B.str() + " rad; #psi^{Rec-A}: " + psi_rec_B.str() + "}";
	pName = "ChargeMapC_Raw_event_" + to_string(i) + "_neutrons_" + to_string(n_neutrons_A) + "_ptNuc_" + to_string(pt_nuc_A) + extension;
	DrawChargeMap(rpd_signal_B_raw, label, folderChargeMap + pName);
      }
      delete rpd_signal_A; delete rpd_signal_B;
      delete rpd_signal_A_raw; delete rpd_signal_B_raw;
      rpd_A.clear();
      rpd_B.clear();
      rpd_A_raw.clear();
      rpd_B_raw.clear();
  }

  std::cout << " drawing plots " << std::endl;
  
  label = "";
  pName = "TotalChargeMapA" + extension;
  //  rpd_signal_A_allEvents->Scale(1/entries);
  DrawChargeMap(rpd_signal_A_allEvents, label, folderChargeMap + pName);
	
  label = "";
  pName = "TotalChargeMapC" + extension;
  DrawChargeMap(rpd_signal_B_allEvents, label, folderChargeMap + pName);
  
  label = "";
  pName = "GenPosMapA" + extension;
  DrawChargeMap(rpd_gen_pos_A_allEvents, label, folderChargeMap + pName);
  label = "";
  pName = "GenPosMapC" + extension;
  DrawChargeMap(rpd_gen_pos_B_allEvents, label, folderChargeMap + pName);

  label = "";
  pName = "GenPosUnitVecMapA" + extension;
  DrawChargeMap(rpd_gen_pos_unit_A_allEvents, label, folderChargeMap + pName);
  label = "";
  pName = "GenPosUnitVecMapC" + extension;
  DrawChargeMap(rpd_gen_pos_unit_B_allEvents, label, folderChargeMap + pName);
  
  //std::cout << " A_1_1: " << rpd_signal_A_allEvents->GetBinContent(1,1) << " B_1_1 " << rpd_signal_B_allEvents->GetBinContent(1,1) << std::endl;


  for (int i = 0; i < 7; ++i) {
    std::cout << " i " << i << std::endl ;
    ostringstream ln, un;
    if (i == 0) {
      un << charge_share_bins.at(i);
    }
    else if (i == 7) {
      ln << charge_share_bins.at(6);
    }
    else {
      ln << charge_share_bins.at(i);
      un << charge_share_bins.at(i+1);
    }
    label = "ZDC, " + ln.str() + " < Charge Fraction <= " + un.str();
    pName = "Gen-Rec_Azimuth_ZDC_ChargeShare" + ln.str() + "_" + un.str() + extension;
    DrawPlot(resoRP_gen_chargeShare.at(i), false, true, "#psi_{0}^{Gen} - #psi_{0}^{Rec} [rad]", "Counts", label, folder1d + pName);
  }
  std::cout << " drawing other plots " << std::endl;
  for(int i = 0; i < (int)resoRP_true_A.size(); i++){
      ostringstream ln, un;
      ln << neutron_bins.at(i); un << neutron_bins.at(i+1);
      label = "ZDC AC, " + ln.str() + " < #bar{n} < " + un.str();
      pName = "RPreso_ZDC_AC_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_averageArms.at(i), false, false, "cos (#psi_{0}^{Rec-A} - #psi_{0}^{Rec-C})", "Counts", label, folder1d + pName);
      label = "ZDC A, " + ln.str() + " < n < " + un.str();
      pName = "Truth-Gen_Azimuth_ZDC_A_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_true_gen.at(i), false, true, "#psi_{0}^{Truth} - #psi_{0}^{Gen-A} [rad]", "Counts", label, folder1d + pName);
      label = "ZDC A, " + ln.str() + " < n < " + un.str();
      pName = "Truth-Rec_Azimuth_ZDC_A_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_true_A.at(i), false, true, "#psi_{0}^{Truth} - #psi_{0}^{Rec-A} [rad]", "Counts", label, folder1d + pName);
      label = "ZDC C, " + ln.str() + " < n < " + un.str();
      pName = "Truth-Rec_Azimuth_ZDC_C_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_true_B.at(i), false, true, "#psi_{0}^{Truth} - #psi_{0}^{Rec-C} [rad]", "Counts", label, folder1d + pName);
      label = "ZDC A, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_Azimuth_ZDC_A_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_gen_A.at(i), false, true, "#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A} [rad]", "Counts", label, folder1d + pName);
      label = "ZDC C, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_Azimuth_ZDC_C_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_gen_B.at(i), false, true, "#psi_{0}^{Gen-C} - #psi_{0}^{Rec-C} [rad]", "Counts", label, folder1d + pName);
      label = "ZDC AC, " + ln.str() + " < n < " + un.str();
      pName = "Truth-Rec_Azimuth_ZDC_AC_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_averageTruth.at(i), false, true, "#psi_{0}^{Truth} - #psi_{0}^{Rec-AC} [rad]", "Counts", label, folder1d + pName);
      label = "ZDC AC, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_Azimuth_ZDC_AC_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRP_averageGen.at(i), false, true, "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC} [rad]", "Counts", label, folder1d + pName);
     
      for (int ptKick = 1; ptKick < 6; ++ptKick) {
	std::stringstream ptKickRange0;
	ptKickRange0 << std::fixed << std::setprecision(3) <<  pt_nuc_intervals[ptKick-1];
	std::stringstream ptKickRange1;
	ptKickRange1 << std::fixed << std::setprecision(3) <<  pt_nuc_intervals[ptKick];
	if (ptKick < 5)	label = "#splitline{ZDC AC                 , " + ln.str() + " < n < " + un.str() + "}{" + ptKickRange0.str() + " <= #it{p}_{T}^{Nuc} < " + ptKickRange1.str()+ "}";
	else label = "#splitline{ZDC AC, " + ln.str() + " < n < " + un.str()+ "}{" +  ptKickRange0.str() + " < #it{p}_{T}^{Nuc}}";
	pName = "Truth-Rec_Azimuth_ZDC_AC_" + ln.str() + "_n_" + un.str() + "_ptNuc_" + std::to_string(ptKick) + extension;
	DrawPlot(resoRP_averageTruth_ptKick[ptKick].at(i), false, true, "#psi_{0}^{Truth} - #psi_{0}^{Rec-AC} [rad]", "Counts", label, folder1d + pName);
      	if (ptKick < 5)	label = "#splitline{ZDC AC, " + ln.str() + " < n < " + un.str() + "}{" + ptKickRange0.str() + " <= #it{p}_{T}^{Nuc} < " + ptKickRange1.str() + "}";
	else label = "#splitline{ZDC AC, " + ln.str() + " < n < " + un.str()+ "}{"  +  ptKickRange0.str() + " < #it{p}_{T}^{Nuc}}";
	pName = "Gen-Rec_Azimuth_ZDC_AC_" + ln.str() + "_n_" + un.str() + "_ptNuc_" + std::to_string(ptKick) + extension;
	DrawPlot(resoRP_averageGen_ptKick[ptKick].at(i), false, true, "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC} [rad]", "Counts", label, folder1d + pName);

	if (ptKick < 5)	label = "#splitline{ZDC A, " + ln.str() + " < n < " + un.str() + "}{" + ptKickRange0.str() + " <= #it{p}_{T}^{Nuc} < " + ptKickRange1.str() + "}";
	else label = "#splitline{ZDC A, " + ln.str() + " < n < " + un.str()+ "}{"  +  ptKickRange0.str() + " < #it{p}_{T}^{Nuc}}";
	pName = "Gen-Rec_Azimuth_ZDC_A_" + ln.str() + "_n_" + un.str() + "_ptNuc_" + std::to_string(ptKick) + extension;
	DrawPlot(resoRP_A_Gen_ptKick[ptKick].at(i), false, true, "#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A} [rad]", "Counts", label, folder1d + pName);

	if (ptKick < 5)	label = "#splitline{ZDC AC                 , " + ln.str() + " < n < " + un.str() + "}{" + ptKickRange0.str() + " <= #it{p}_{T}^{Nuc} < " + ptKickRange1.str()+ "}";
	else label = "#splitline{ZDC AC, " + ln.str() + " < n < " + un.str()+ "}{" +  ptKickRange0.str() + " < #it{p}_{T}^{Nuc}}";
	pName = "RPreso_ZDC_AC_" + ln.str() + "_n_" + un.str() + "_ptNuc_" + std::to_string(ptKick) + extension;
	DrawPlot(resoRP_averageArms_ptKick[ptKick].at(i), false, false, "cos (#psi_{0}^{Rec-A} - #psi_{0}^{Rec-C})", "Counts", label, folder1d + pName);

	if (ptKick < 5)	label = "#splitline{ZDC A, " + ln.str() + " < n < " + un.str() + "}{" + ptKickRange0.str() + " <= #it{p}_{T}^{Nuc} < " + ptKickRange1.str() + "}";
	else label = "#splitline{ZDC A, " + ln.str() + " < n < " + un.str()+ "}{"  +  ptKickRange0.str() + " < #it{p}_{T}^{Nuc}}";
	pName = "Gen-Rec_COM_Azimuth_ZDC_A_" + ln.str() + "_n_" + un.str() + "_ptNuc_" + std::to_string(ptKick) + extension;
	DrawPlot(resoRP_COM_A_Gen_ptKick[ptKick].at(i), false, true, "#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A} [rad]", "Counts", label, folder1d + pName);

      }
      
      // X POSITION
      label = "ZDC A, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_X_ZDC_A_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoX_gen_A.at(i), false, true, "Q_{x}^{Gen-A} - Q_{x}^{Rec-A} [mm]", "Counts", label, folder1d + pName);
      label = "ZDC C, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_X_ZDC_C_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoX_gen_B.at(i), false, true, "Q_{x}^{Gen-C} - Q_{x}^{Rec-C} [mm]", "Counts", label, folder1d + pName);
      // Y POSITION
      label = "ZDC A, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_Y_ZDC_A_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoY_gen_A.at(i), false, true, "Q_{y}^{Gen-A} - Q_{y}^{Rec-A} [mm]", "Counts", label, folder1d + pName);
      label = "ZDC C, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_Y_ZDC_C_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoY_gen_B.at(i), false, true, "Q_{y}^{Gen-C} - Q_{y}^{Rec-C} [mm]", "Counts", label, folder1d + pName);
      //R
      label = "ZDC A, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_R_ZDC_A_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRadius_gen_A.at(i), false, true, "|Q|^{Gen-A} - |Q|^{Rec-A} [mm]", "Counts", label, folder1d + pName);
      label = "ZDC C, " + ln.str() + " < n < " + un.str();
      pName = "Gen-Rec_R_ZDC_C_" + ln.str() + "_n_" + un.str() + extension;
      DrawPlot(resoRadius_gen_B.at(i), false, true, "|Q|^{Gen-C} - |Q|^{Rec-C} [mm]", "Counts", label, folder1d + pName);
  }


  for(int i = 0; i < (int)resoRP_true_vs_RP_true.size(); i++){
      ostringstream ln, un;
      ln << neutron_bins.at(i); un << neutron_bins.at(i+1);
      label = "ZDC AC, " + ln.str() + " < n < " + un.str();
      pName = "2D_Truth-Rec_vs_Truth_Azimuth_ZDC_AC_" + ln.str() + "_n_" + un.str() + extension;
      Draw2DPlot(resoRP_true_vs_RP_true.at(i), "COLZ", false, "#psi_{0}^{Truth} [rad]", "#psi_{0}^{Truth} - #psi_{0}^{Rec-AC} [rad]", label, folder2d + pName);
      label = "ZDC AC, " + ln.str() + " < n < " + un.str();
      pName = "2D_Gen-Rec_vs_Truth_Azimuth_ZDC_AC_" + ln.str() + "_n_" + un.str() + extension;
      Draw2DPlot(resoRP_gen_vs_RP_true.at(i), "COLZ", false, "#psi_{0}^{Truth} [rad]", "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC} [rad]", label, folder2d + pName);
      label = "ZDC AC, " + ln.str() + " < n < " + un.str();
      pName = "2D_Gen-Rec_vs_Gen_Azimuth_ZDC_AC_" + ln.str() + "_n_" + un.str() + extension;
      Draw2DPlot(resoRP_gen_vs_RP_gen.at(i), "COLZ", false, "#psi_{0}^{Gen-AC} [rad]", "#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC} [rad]", label, folder2d + pName);
      label = "ZDC AC, " + ln.str() + " < #bar{n} < " + un.str();
      pName = "2D_QxA_vs_QxC_ZDC_" + ln.str() + "_n_" + un.str() + extension;
      Draw2DPlot(QxA_vs_QxC.at(i), "COLZ", false, "Q_{x}^{Rec-A} [mm]", "Q_{x}^{Rec-C} [mm]", label, folder2d + pName);
      label = "ZDC AC, " + ln.str() + " < #bar{n} < " + un.str();
      pName = "2D_QyA_vs_QyC_ZDC_" + ln.str() + "_n_" + un.str() + extension;
      Draw2DPlot(QyA_vs_QyC.at(i), "COLZ", false, "Q_{y}^{Rec-A} [mm]", "Q_{y}^{Rec-C} [mm]", label, folder2d + pName);
      label = "ZDC AC, " + ln.str() + " < #bar{n} < " + un.str();
      pName = "2D_Gen_QxA_vs_Gen_QxC_ZDC_" + ln.str() + "_n_" + un.str() + extension;
      Draw2DPlot(gen_QxA_vs_QxC.at(i), "COLZ", false, "Q_{x}^{Gen-A} [mm]", "Q_{x}^{Gen-C} [mm]", label, folder2d + pName);
      label = "ZDC AC, " + ln.str() + " < #bar{n} < " + un.str();
      pName = "2D_Gen_QyA_vs_Gen_QyC_ZDC_" + ln.str() + "_n_" + un.str() + extension;
      Draw2DPlot(gen_QyA_vs_QyC.at(i), "COLZ", false, "Q_{y}^{Gen-A} [mm]", "Q_{y}^{Gen-C} [mm]", label, folder2d + pName);
  }


  //Angle
  pName = "RPreso_ZDC_AC_AllNeutronClasses" + extension;
  StackNeutronBins( resoRP_averageArms, "", folderMainRes + pName , false );
  pName = "Truth-Gen_Azimuth_ZDC_A_AllNeutronClasses" + extension;
  StackNeutronBins( resoRP_true_gen, "", folderMainRes + pName );
  pName = "Truth-Rec_Azimuth_ZDC_A_AllNeutronClasses" + extension;
  StackNeutronBins( resoRP_true_A, "", folderMainRes + pName );
  pName = "Truth-Rec_Azimuth_ZDC_C_AllNeutronClasses" + extension;
  StackNeutronBins( resoRP_true_B, "", folderMainRes + pName );
  pName = "Gen-Rec_Azimuth_ZDC_A_AllNeutronClasses" + extension;
  StackNeutronBins( resoRP_gen_A, "", folderMainRes + pName );
  pName = "Gen-Rec_Azimuth_ZDC_C_AllNeutronClasses" + extension;
  StackNeutronBins( resoRP_gen_B, "", folderMainRes + pName );
  pName = "Truth-Rec_Azimuth_ZDC_AC_AllNeutronClasses"  + extension;
  StackNeutronBins( resoRP_averageTruth, "", folderMainRes + pName );
  pName = "Gen-Rec_Azimuth_ZDC_AC_AllNeutronClasses" + extension;
  StackNeutronBins( resoRP_averageGen, "", folderMainRes + pName );
  
  for (int ptKick = 1; ptKick < 6; ++ptKick) {
    pName = "Truth-Rec_Azimuth_ZDC_AC_AllNeutronClasses_ptKick" + std::to_string(ptKick) + extension;
    StackNeutronBins( resoRP_averageTruth_ptKick[ptKick], "", folderMainRes + pName);
    pName = "Gen-Rec_Azimuth_ZDC_AC_AllNeutronClasses_ptKick" + std::to_string(ptKick) + extension;
    StackNeutronBins( resoRP_averageGen_ptKick[ptKick], "", folderMainRes + pName);
    pName = "Gen-Rec_Azimuth_ZDC_A_AllNeutronClasses_ptKick" + std::to_string(ptKick) + extension;
    StackNeutronBins( resoRP_A_Gen_ptKick[ptKick], "", folderMainRes + pName);
    pName = "Gen-Rec_COM_Azimuth_ZDC_A_AllNeutronClasses_ptKick" + std::to_string(ptKick) + extension;
    StackNeutronBins( resoRP_COM_A_Gen_ptKick[ptKick], "", folderMainRes + pName);
  }
  //X position
  pName = "Gen-Rec_X_ZDC_A_AllNeutronClasses" + extension;
  StackNeutronBins( resoX_gen_A, "", folderMainRes + pName );
  pName = "Gen-Rec_X_ZDC_C_AllNeutronClasses" + extension;
  StackNeutronBins( resoX_gen_B, "", folderMainRes + pName );
  //Y position
  pName = "Gen-Rec_Y_ZDC_A_AllNeutronClasses" + extension;
  StackNeutronBins( resoY_gen_A, "", folderMainRes + pName );
  pName = "Gen-Rec_Y_ZDC_C_AllNeutronClasses" + extension;
  StackNeutronBins( resoY_gen_B, "", folderMainRes + pName );
  //R
  pName = "Gen-Rec_R_ZDC_A_AllNeutronClasses" + extension;
  StackNeutronBins( resoRadius_gen_A, "", folderMainRes + pName );
  pName = "Gen-Rec_R_ZDC_C_AllNeutronClasses" + extension;
  StackNeutronBins( resoRadius_gen_B, "", folderMainRes + pName );

  pName = "Gen-Rec_Azimuth_ZDC_Reso_vs_ChargeShare" + extension;
  //  PlotChargeShareDependence(resoRP_gen_chargeShare, "#sigma_{#psi_{0}^{Gen} - #psi_{0}^{Rec}} [rad]", folderChargeShareDep + pName);

  TFile* resoFile = new TFile("ResoFileFermi.root","RECREATE");

  resoFile->cd();
  //Neutron dependence
  TGraphErrors* tge_resoRP_averageArms_ptKick[5];
  TGraphErrors* tge_resoRP_averageTruth_ptKick[5];
  TGraphErrors* tge_resoRP_averageGen_ptKick[5];
  TGraphErrors* tge_resoRP_A_Gen_ptKick[5];
  TGraphErrors* tge_resoRP_A_Truth_ptKick[5];
  TGraphErrors* tge_resoRP_COM_A_Gen_ptKick[5];
  for (int ptKick = 1; ptKick < 6; ++ptKick) {
    
    pName = "RPResolution_ZDC_AC_Reso_ptKick_vs_nClass_ptKick_" + std::to_string(ptKick) + extension;
    tge_resoRP_averageArms_ptKick[ptKick-1] = PlotNeutronDependenceMean2( resoRP_averageArms_ptKick[ptKick], "#sqrt{<cos (#psi_{0}^{Rec-A} - #psi_{0}^{Rec-C})>}", folderNeutronDep + pName, ptKick );
    pName = "Truth-Rec_Azimuth_ZDC_AC_AllNeutronClasses_ptKick_" + std::to_string(ptKick)  + extension;
    tge_resoRP_averageTruth_ptKick[ptKick-1] = PlotNeutronDependence2( resoRP_averageTruth_ptKick[ptKick], "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Rec-AC}} [rad]", folderNeutronDep + pName);
    pName = "Gen-Rec_Azimuth_ZDC_AC_AllNeutronClasses_ptKick_" + std::to_string(ptKick) + extension;
    tge_resoRP_averageGen_ptKick[ptKick-1] = PlotNeutronDependence2( resoRP_averageGen_ptKick[ptKick], "#sigma_{#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC}} [rad]", folderNeutronDep + pName );

    pName = "Gen-Rec_Azimuth_ZDC_A_AllNeutronClasses_ptKick_" + std::to_string(ptKick) + extension;
    tge_resoRP_A_Gen_ptKick[ptKick-1] = PlotNeutronDependence2( resoRP_A_Gen_ptKick[ptKick], "#sigma_{#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName );
    pName = "Truth-Rec_Azimuth_ZDC_A_AllNeutronClasses_ptKick_" + std::to_string(ptKick) + extension;
    tge_resoRP_A_Truth_ptKick[ptKick-1] = PlotNeutronDependence2( resoRP_A_Truth_ptKick[ptKick], "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName );

    pName = "Gen-Rec_COM_Azimuth_ZDC_A_AllNeutronClasses_ptKick_" + std::to_string(ptKick) + extension;
    tge_resoRP_COM_A_Gen_ptKick[ptKick-1] = PlotNeutronDependence2( resoRP_COM_A_Gen_ptKick[ptKick], "#sigma_{#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName );
  }

  gStyle->SetPalette(kRainBow);
  
  pName = "RPResolution_ZDC_AC_Reso_vs_nClass_MultiGraph" + extension;
  PlotNeutronDependence_MultiGraph(tge_resoRP_averageArms_ptKick , "#sqrt{<cos (#psi_{0}^{Rec-A} - #psi_{0}^{Rec-C})>}", folderNeutronDep + pName, pt_nuc_intervals, 1.5 );
  pName = "Truth-Rec_Azimuth_ZDC_AC_AllNeutronClasses_MultiGraph"  + extension;
  PlotNeutronDependence_MultiGraph(tge_resoRP_averageTruth_ptKick , "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Rec-C}} [rad]", folderNeutronDep + pName , pt_nuc_intervals, 2.8);
  pName = "Gen-Rec_Azimuth_ZDC_AC_AllNeutronClasses_MultiGraph" + extension;
  PlotNeutronDependence_MultiGraph(tge_resoRP_averageGen_ptKick , "#sigma_{#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC}} [rad]", folderNeutronDep + pName , pt_nuc_intervals, 1.4);


  pName = "Gen-Rec_Azimuth_ZDC_A_AllNeutronClasses_MultiGraph" + extension;
  PlotNeutronDependence_MultiGraph(tge_resoRP_A_Gen_ptKick , "#sigma_{#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName , pt_nuc_intervals, 1.4);
  pName = "Truth-Rec_Azimuth_ZDC_A_AllNeutronClasses_MultiGraph"  + extension;
  PlotNeutronDependence_MultiGraph(tge_resoRP_A_Truth_ptKick , "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName , pt_nuc_intervals, 2.8);

  pName = "Gen-Rec_COM_Azimuth_ZDC_A_AllNeutronClasses_MultiGraph" + extension;
  PlotNeutronDependence_MultiGraph(tge_resoRP_COM_A_Gen_ptKick , "#sigma_{#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName , pt_nuc_intervals, 1.4);
  
  pName = "RPResolution_ZDC_AC_Reso_vs_nClass" + extension;
  PlotNeutronDependenceMean( resoRP_averageArms, "#sqrt{<cos (#psi_{0}^{Rec-A} - #psi_{0}^{Rec-C})>}", folderNeutronDep + pName );
  pName = "Truth-Gen_Azimuth_ZDC_A_Reso_vs_nClass" + extension;
  PlotNeutronDependence( resoRP_true_gen, "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Gen-A}} [rad]", folderNeutronDep + pName );
  pName = "Truth-Rec_Azimuth_ZDC_A_Reso_vs_nClass" + extension;
  PlotNeutronDependence( resoRP_true_A, "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName );
  pName = "Truth-Rec_Azimuth_ZDC_C_Reso_vs_nClass" + extension;
  PlotNeutronDependence( resoRP_true_B, "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Rec-C}} [rad]", folderNeutronDep + pName );
  pName = "Gen-Rec_Azimuth_ZDC_A_Reso_vs_nClass" + extension;
  PlotNeutronDependence( resoRP_gen_A, "#sigma_{#psi_{0}^{Gen-A} - #psi_{0}^{Rec-A}} [rad]", folderNeutronDep + pName );
  pName = "Gen-Rec_Azimuth_ZDC_C_Reso_vs_nClass" + extension;
  PlotNeutronDependence( resoRP_gen_B, "#sigma_{#psi_{0}^{Gen-C} - #psi_{0}^{Rec-C}} [rad]", folderNeutronDep + pName );
  pName = "Truth-Rec_Azimuth_ZDC_AC_AllNeutronClasses"  + extension;
  PlotNeutronDependence( resoRP_averageTruth, "#sigma_{#psi_{0}^{Truth} - #psi_{0}^{Rec-AC}} [rad]", folderNeutronDep + pName );
  pName = "Gen-Rec_Azimuth_ZDC_AC_AllNeutronClasses" + extension;
  PlotNeutronDependence( resoRP_averageGen, "#sigma_{#psi_{0}^{Gen-AC} - #psi_{0}^{Rec-AC}} [rad]", folderNeutronDep + pName );

  //X position
  pName = "Gen-Rec_X_ZDC_A_AllNeutronClasses" + extension;
  PlotNeutronDependence( resoX_gen_A, "#sigma_{Q_{x}^{Gen-A} - Q_{x}^{Rec-A}} [mm]", folderNeutronDep + pName );
  pName = "Gen-Rec_X_ZDC_C_AllNeutronClasses" + extension;
  PlotNeutronDependence( resoX_gen_B, "#sigma_{Q_{x}^{Gen-C} - Q_{x}^{Rec-C}} [mm]", folderNeutronDep + pName );
  //Y position
  pName = "Gen-Rec_Y_ZDC_A_AllNeutronClasses" + extension;
  PlotNeutronDependence( resoY_gen_A, "#sigma_{Q_{y}^{Gen-A} - Q_{y}^{Rec-A}} [mm]", folderNeutronDep + pName );
  pName = "Gen-Rec_Y_ZDC_C_AllNeutronClasses" + extension;
  PlotNeutronDependence( resoY_gen_B, "#sigma_{Q_{y}^{Gen-C} - Q_{y}^{Rec-C}} [mm]", folderNeutronDep + pName );
  //R
  pName = "Gen-Rec_R_ZDC_A_AllNeutronClasses" + extension;
  PlotNeutronDependence( resoRadius_gen_A, "#sigma_{|Q^{Gen-A}| - |Q^{Rec-A}|} [mm]", folderNeutronDep + pName );
  pName = "Gen-Rec_R_ZDC_C_AllNeutronClasses" + extension;
  PlotNeutronDependence( resoRadius_gen_B, "#sigma_{|Q^{Gen-C}| - |Q^{Rec-C}|} [mm]", folderNeutronDep + pName );



  for (int ptKick = 1; ptKick < 6; ++ptKick) {
    delete tge_resoRP_averageArms_ptKick[ptKick-1];
    delete tge_resoRP_averageTruth_ptKick[ptKick-1];
    delete tge_resoRP_averageGen_ptKick[ptKick-1];
  }

}


void PlotNeutronDependence_MultiGraph(TGraphErrors *tge[5], string yAxis, string outLabel, double pt_nuc_intervals[5], double yMax){

  TMultiGraph *mg = new TMultiGraph();

  TCanvas* c1 = new TCanvas("c1","",550,500);
  c1->cd();
  
  for (int i = 0; i < 5; ++i) {
    mg->Add(tge[i],"PM");
  }
  mg->Draw("A PMC PLC");

  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.2);
  //  gPad->SetBottomMargin(0.15);
  double yMin = 0.;
  mg->GetHistogram()->GetXaxis()->SetRangeUser(20,40);
  mg->GetHistogram()->GetYaxis()->SetRangeUser(yMin, yMax);
  mg->GetHistogram()->GetXaxis()->SetTitle("N_{neutrons}");
  mg->GetHistogram()->GetYaxis()->SetTitle(yAxis.c_str());
  mg->GetHistogram()->GetYaxis()->SetTitleSize(0.055);
  mg->GetHistogram()->GetXaxis()->SetTitleOffset(1.25);
  mg->GetHistogram()->GetYaxis()->SetTitleOffset(1.6);
  TLine *l1 = new TLine();
  l1->SetLineStyle(kDashed);
  l1->SetLineColor(kBlack);
  for(int i = 1; i < neutron_bins.size()-1; i++){
    l1->DrawLine(neutron_bins.at(i), yMin, neutron_bins.at(i), yMax);
  }
  char label[256];
  TLegend *leg = new TLegend(0.5, 0.62, 0.9, 0.92, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillColor(0);
  leg->SetTextFont(42);
  //  leg->SetTextSize(-1);
  sprintf(label,"%0.0f <= #it{p}_{T}^{Nuc} < %0.0f MeV", pt_nuc_intervals[0]*1e3,pt_nuc_intervals[1]*1e3);
  leg->AddEntry(tge[0],label,"plfe");
  sprintf(label,"%0.0f <= #it{p}_{T}^{Nuc} < %0.0f MeV", pt_nuc_intervals[1]*1e3,pt_nuc_intervals[2]*1e3);
  leg->AddEntry(tge[1],label,"plfe");
  sprintf(label,"%0.0f <= #it{p}_{T}^{Nuc} < %0.0f MeV", pt_nuc_intervals[2]*1e3,pt_nuc_intervals[3]*1e3);
  leg->AddEntry(tge[2],label,"plfe");
  sprintf(label,"%0.0f <= #it{p}_{T}^{Nuc} < %0.0f MeV", pt_nuc_intervals[3]*1e3,pt_nuc_intervals[4]*1e3);
  leg->AddEntry(tge[3],label,"plfe");
  sprintf(label,"%0.0f <= #it{p}_{T}^{Nuc}", pt_nuc_intervals[4]*1e3);
  leg->AddEntry(tge[4],label,"plfe");
  leg->Draw();
  c1->Print(outLabel.c_str());
  delete c1;
  return tge;

  //TGraph* hNDep = new TGra
}



TGraphErrors* PlotChargeShareDependence( vector < TH1D* > h1, string yAxis, string outLabel, int parameter){

  TH1D* hbase = new TH1D("base","base",(int)h1.size(),0,1);
  string n_class;
  for(int i = 0; i < charge_share_bins.size()-1; i++){
    ostringstream nl, nu;
    n_class = "[" + nl.str() + ";" + nu.str() + "]";
    nl << charge_share_bins.at(i);
    nu << charge_share_bins.at(i+1);
    hbase->GetXaxis()->SetBinLabel(i+1,n_class.c_str());
  }
  double x[(int)h1.size()], ex[(int)h1.size()], y[(int)h1.size()], ey[(int)h1.size()];
  for(int i = 0; i < (int)h1.size(); i++){
    x[i] = (charge_share_bins.at(i)+charge_share_bins.at(i+1))/2;
    ex[i] = 0.;
    GaussianFitGet(h1.at(i),parameter,&y[i],&ey[i]);
    std::cout << " CHARGE SHARE bin " << i << " x " << x[i] <<  " resolution " << y[i] << std::endl;
  }

  TGraphErrors* tge = new TGraphErrors((int)h1.size(), x,y,ex,ey);
  TCanvas* c1 = new TCanvas("c1","",550,500);
  c1->cd();
  tge->SetMarkerColor(kBlue+1);
  tge->SetLineColor(kBlue+1);
  tge->SetMarkerStyle(21);
  tge->GetYaxis()->SetTitle(yAxis.c_str());
  tge->GetYaxis()->SetRangeUser(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetLimits(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetTitleSize(0.055);
  tge->GetXaxis()->SetTitleOffset(1.25);
  tge->GetYaxis()->SetTitleOffset(1.6);
  tge->GetXaxis()->SetRangeUser(0,1);
  tge->GetXaxis()->SetTitle("Charge Share Fraction in Largest Channel");
  tge->Draw("AP");

  //  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.20);
  //  gPad->SetBottomMargin(0.13);

  TLine *l1 = new TLine();
  l1->SetLineStyle(kDashed);
  l1->SetLineColor(kBlack);
  for(int i = 1; i < charge_share_bins.size()-1; i++){
    l1->DrawLine(charge_share_bins.at(i),tge->GetYaxis()->GetXmin(), charge_share_bins.at(i), tge->GetYaxis()->GetXmax() );
  }

  c1->Print(outLabel.c_str());
  delete c1;
  return tge;

  //TGraph* hNDep = new TGra
}



void PlotNeutronDependence( vector < TH1D* > h1, string yAxis, string outLabel, int parameter){

  TH1D* hbase = new TH1D("base","base",(int)h1.size(),20,40);
  string n_class;
  for(int i = 0; i < neutron_bins.size()-1; i++){
    ostringstream nl, nu;
    n_class = "[" + nl.str() + ";" + nu.str() + "]";
    nl << neutron_bins.at(i);
    nu << neutron_bins.at(i+1);
    hbase->GetXaxis()->SetBinLabel(i+1,n_class.c_str());
  }
  double x[(int)h1.size()], ex[(int)h1.size()], y[(int)h1.size()], ey[(int)h1.size()];
  for(int i = 0; i < (int)h1.size(); i++){
    x[i] = (neutron_bins.at(i)+neutron_bins.at(i+1))/2;
    ex[i] = 0.;
    GaussianFitGet(h1.at(i),parameter,&y[i],&ey[i]);
  }

  TGraphErrors* tge = new TGraphErrors((int)h1.size(), x,y,ex,ey);
  TCanvas* c1 = new TCanvas("c1","",550,500);
  c1->cd();
  tge->SetMarkerColor(kBlue+1);
  tge->SetLineColor(kBlue+1);
  tge->SetMarkerStyle(21);
  tge->GetYaxis()->SetTitle(yAxis.c_str());
  tge->GetYaxis()->SetRangeUser(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetLimits(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetTitleSize(0.055);
  tge->GetXaxis()->SetTitleOffset(1.25);
  tge->GetYaxis()->SetTitleOffset(1.6);
  tge->GetXaxis()->SetRangeUser(20,40);
  tge->GetXaxis()->SetLimits(20,40);
  tge->GetXaxis()->SetTitle("N_{neutrons}");
  tge->Draw("AP");

  //  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.20);
  //  gPad->SetBottomMargin(0.13);

  TLine *l1 = new TLine();
  l1->SetLineStyle(kDashed);
  l1->SetLineColor(kBlack);
  for(int i = 1; i < neutron_bins.size()-1; i++){
    l1->DrawLine(neutron_bins.at(i),tge->GetYaxis()->GetXmin(), neutron_bins.at(i), tge->GetYaxis()->GetXmax() );
  }

  c1->Print(outLabel.c_str());
  delete c1;
  return tge;

  //TGraph* hNDep = new TGra
}


TGraphErrors* PlotNeutronDependence2( vector < TH1D* > h1, string yAxis, string outLabel, int parameter){

  TH1D* hbase = new TH1D("base","base",(int)h1.size(),20,40);
  string n_class;
  for(int i = 0; i < neutron_bins.size()-1; i++){
    ostringstream nl, nu;
    n_class = "[" + nl.str() + ";" + nu.str() + "]";
    nl << neutron_bins.at(i);
    nu << neutron_bins.at(i+1);
    hbase->GetXaxis()->SetBinLabel(i+1,n_class.c_str());
  }
  double x[(int)h1.size()], ex[(int)h1.size()], y[(int)h1.size()], ey[(int)h1.size()];
  for(int i = 0; i < (int)h1.size(); i++){
    x[i] = (neutron_bins.at(i)+neutron_bins.at(i+1))/2;
    ex[i] = 0.;
    GaussianFitGet(h1.at(i),parameter,&y[i],&ey[i]);
  }

  TGraphErrors* tge = new TGraphErrors((int)h1.size(), x,y,ex,ey);
  TCanvas* c1 = new TCanvas("c1","",550,500);
  c1->cd();
  tge->SetMarkerColor(kBlue+1);
  tge->SetLineColor(kBlue+1);
  tge->SetMarkerStyle(21);
  tge->GetYaxis()->SetTitle(yAxis.c_str());
  tge->GetYaxis()->SetRangeUser(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetLimits(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetTitleSize(0.055);
  tge->GetXaxis()->SetTitleOffset(1.25);
  tge->GetYaxis()->SetTitleOffset(1.6);
  tge->GetXaxis()->SetRangeUser(20,40);
  tge->GetXaxis()->SetLimits(20,40);
  tge->GetXaxis()->SetTitle("N_{neutrons}");
  tge->Draw("AP");

  //  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.20);
  //  gPad->SetBottomMargin(0.13);

  TLine *l1 = new TLine();
  l1->SetLineStyle(kDashed);
  l1->SetLineColor(kBlack);
  for(int i = 1; i < neutron_bins.size()-1; i++){
    l1->DrawLine(neutron_bins.at(i),tge->GetYaxis()->GetXmin(), neutron_bins.at(i), tge->GetYaxis()->GetXmax() );
  }

  c1->Print(outLabel.c_str());
  delete c1;
  return tge;

  //TGraph* hNDep = new TGra
}



void PlotNeutronDependenceMean( vector < TH1D* > h1, string yAxis, string outLabel){

  TH1D* hbase = new TH1D("base","base",(int)h1.size(),20,40);
  double x[(int)h1.size()], ex[(int)h1.size()], y[(int)h1.size()], ey[(int)h1.size()];
  for(int i = 0; i < (int)h1.size(); i++){
    x[i] = (neutron_bins.at(i)+neutron_bins.at(i+1))/2;
    ex[i] = 0.;
    y[i] = sqrt(h1.at(i)->GetMean());
    ey[i] = (0.5)*(y[i]/sqrt(h1.at(i)->GetMean()))*h1.at(i)->GetMeanError();
  }

  TGraphErrors* tge = new TGraphErrors((int)h1.size(), x,y,ex,ey);
  TCanvas* c1 = new TCanvas("c1","",550,500);
  c1->cd();
  tge->SetMarkerColor(kBlue+1);
  tge->SetLineColor(kBlue+1);
  tge->SetMarkerStyle(21);
  tge->GetYaxis()->SetTitle(yAxis.c_str());
  tge->GetYaxis()->SetRangeUser(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetLimits(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetTitleSize(0.055);
  tge->GetXaxis()->SetTitleOffset(1.25);
  tge->GetYaxis()->SetTitleOffset(1.6);
  tge->GetXaxis()->SetRangeUser(20,40);
  tge->GetXaxis()->SetLimits(20,40);
  tge->GetXaxis()->SetTitle("N_{neutrons}");
  tge->Draw("AP");

  //  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.20);
  //  gPad->SetBottomMargin(0.13);

  tge->Write("Resolution_vs_n");
  TLine *l1 = new TLine();
  l1->SetLineStyle(kDashed);
  l1->SetLineColor(kBlack);
  for(int i = 1; i < neutron_bins.size()-1; i++){
    l1->DrawLine(neutron_bins.at(i),tge->GetYaxis()->GetXmin(), neutron_bins.at(i), tge->GetYaxis()->GetXmax() );
  }

  c1->Print(outLabel.c_str());
  c1->Write();
  delete c1;
  return tge;

  //TGraph* hNDep = new TGra
}


TGraphErrors* PlotNeutronDependenceMean2( vector < TH1D* > h1, string yAxis, string outLabel, int ptKick){

  TH1D* hbase = new TH1D("base","base",(int)h1.size(),20,40);
  double x[(int)h1.size()], ex[(int)h1.size()], y[(int)h1.size()], ey[(int)h1.size()];
  for(int i = 0; i < (int)h1.size(); i++){
    x[i] = (neutron_bins.at(i)+neutron_bins.at(i+1))/2;
    ex[i] = 0.;
    y[i] = sqrt(h1.at(i)->GetMean());
    ey[i] = (0.5)*(y[i]/sqrt(h1.at(i)->GetMean()))*h1.at(i)->GetMeanError();
  }

  TGraphErrors* tge = new TGraphErrors((int)h1.size(), x,y,ex,ey);
  TCanvas* c1 = new TCanvas("c1","",550,500);
  c1->cd();
  tge->SetMarkerColor(kBlue+1);
  tge->SetLineColor(kBlue+1);
  tge->SetMarkerStyle(21);
  tge->GetYaxis()->SetTitle(yAxis.c_str());
  tge->GetYaxis()->SetRangeUser(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetLimits(tge->GetYaxis()->GetXmin()*0.9,tge->GetYaxis()->GetXmax()*1.1);
  tge->GetYaxis()->SetTitleSize(0.055);
  tge->GetXaxis()->SetTitleOffset(1.25);
  tge->GetYaxis()->SetTitleOffset(1.6);
  tge->GetXaxis()->SetRangeUser(20,40);
  tge->GetXaxis()->SetLimits(20,40);
  tge->GetXaxis()->SetTitle("N_{neutrons}");
  tge->Draw("AP");

  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.20);
  //  gPad->SetBottomMargin(0.13);

  tge->Write(Form("Resolution_vs_n_ptKick%d",ptKick));
  TLine *l1 = new TLine();
  l1->SetLineStyle(kDashed);
  l1->SetLineColor(kBlack);
  for(int i = 1; i < neutron_bins.size()-1; i++){
    l1->DrawLine(neutron_bins.at(i),tge->GetYaxis()->GetXmin(), neutron_bins.at(i), tge->GetYaxis()->GetXmax() );
  }

  c1->Print(outLabel.c_str());
  c1->Write();
  delete c1;
  return tge;

  //TGraph* hNDep = new TGra
}

void StackNeutronBins( vector < TH1D* > h1, string label, string outputName, bool fit  ){

  label = "";
  TH1D* clone = (TH1D*)h1.at(0)->Clone();
  for(int i = 1; i < (int)h1.size(); i++){
    clone->Add(h1.at(i));
  }
  DrawPlot(clone, false, fit, "keep", "keep", label, outputName);
  return;
}

void StackNeutronBinsPtKick( vector < TH1D* > h1, string label, string outputName, int ptKick, bool fit){

  label = "";
  TH1D* clone = (TH1D*)h1.at(0)->Clone();
  for(int i = 1; i < (int)h1.size(); i++){
    clone->Add(h1.at(i));
  }
  DrawPlot(clone, false, fit, "keep", "keep", label, outputName);
  return;
}

double makeAngleDiff( double a, double b){
    double diff = a - b;
    if( diff < -TMath::Pi() ){ diff += TMath::TwoPi(); }
    if( diff > TMath::Pi() ) { diff += -TMath::TwoPi(); }
    return diff;

}

int FindNeutronBin( double n_neutrons ){
    for(int i = 0; i < neutron_bins.size()-1; i++){
      if( n_neutrons >= neutron_bins.at(i) && n_neutrons < neutron_bins.at(i+1))
        return i;
    }
    return -1;
}

int FindChargeShareBin( vector<double> rpdSignal ){
  double maxSignal = -1;
  double totalSignal = 0;
  double chargeShare = -1;
  for(int i = 0; i < 16; i++){
    if (rpdSignal[i] > maxSignal) maxSignal = rpdSignal[i];
    totalSignal += rpdSignal[i];
  }
  chargeShare = maxSignal / totalSignal;
  for(int i = 0; i < 7; i++){
    if (chargeShare > charge_share_bins[i] && chargeShare <= charge_share_bins[i+1]) return i;
  }
  return -1;
}


void Draw2DPlot(TH2D* h2, string draw_options, bool logz, string x_title,
                string y_axis, string label, string out_file)
{
  TCanvas* c1 = new TCanvas(x_title.c_str(),"",600,500);
  c1->cd();
  h2->SetMarkerSize(0.5);
  h2->SetMarkerStyle(20);
  if(x_title != "keep") h2->GetXaxis()->SetTitle(x_title.c_str());
  h2->GetXaxis()->SetTitleSize(0.05);
  if(y_axis != "keep") h2->GetYaxis()->SetTitle(y_axis.c_str());
  h2->GetYaxis()->SetTitleSize(0.05);
  h2->Draw(draw_options.c_str());
  if(logz) {
    gPad->SetLogz();
    std::cout << h2->GetMinimum(1e-8) << std::endl;
    h2->GetZaxis()->SetRangeUser( h2->GetMinimum(1e-8),h2->GetMaximum()*2);
  }
  h2->GetXaxis()->SetTitleOffset(1.25);
  h2->GetYaxis()->SetTitleOffset(1.25);

  // by default: left (0.15), right (0.15), top (0.05), bottom (0.15)
  //  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.18);
  //  gPad->SetLeftMargin(0.15);
  //  gPad->SetBottomMargin(0.15);
  
  TLatex *lat = new TLatex();
  lat->SetTextFont(72);
  lat->SetTextSize(0.04);
  lat->SetTextFont(42);
  lat->DrawLatexNDC(.55,.9,label.c_str());

  gStyle->SetOptStat(0);
  c1->Print(out_file.c_str());
  gStyle->SetOptStat("rme");
  delete c1;
}

void DrawPlot(TH1D* h1, bool logy, bool fit, string x_title, string y_axis, string label, string out_file)
{

  TCanvas* c1 = new TCanvas(x_title.c_str(),"",550,500);
  c1->cd();

  h1->SetLineColor(kAzure+1);
  h1->SetFillColorAlpha(kAzure,0.3);
  if( x_title != "keep") h1->GetXaxis()->SetTitle(x_title.c_str());
  h1->GetXaxis()->SetTitleSize(0.05);
  h1->GetXaxis()->SetTitleOffset(1.25);
  h1->GetYaxis()->SetTitleOffset(1.45);
  if( y_axis != "keep") h1->GetYaxis()->SetTitle(y_axis.c_str());
  h1->GetYaxis()->SetTitleSize(0.05);
   gStyle->SetOptStat("rme");
  if( fit ){
    gStyle->SetOptStat("e");
    GaussianFit( h1 );
  }
  h1->Draw();

  if(logy){
    gPad->SetLogy();
     h1->GetYaxis()->SetRangeUser(1e-6, h1->GetMaximum()*100.);
  }
  /*if(h1.size() > 1){
    for(int i = 1; i < h1.size(); i++){
      h1.at(i)->Draw("SAME");
      leg->AddEntry(h1.at(i),h1.at(i)->GetTitle(),"l");
    }
  }*/
  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.15);
  gPad->SetBottomMargin(0.15);

  TLatex *lat = new TLatex();
  lat->SetTextFont(72);
  lat->SetTextSize(0.04);
  lat->SetTextFont(42);
  lat->DrawLatexNDC(.2,0.86,label.c_str());
  //leg->SetTextSize(0.035  );
  //leg->Draw();
  gPad->Update();
  /*
  TPaveStats *st = (TPaveStats*)h1->FindObject("stats");
  st->SetX1NDC(0.5);
  st->SetX2NDC(0.95);
  st->SetY1NDC(0.85);
  st->SetY2NDC(0.99);
  */
  gPad->Update();
  c1->Print(out_file.c_str());
  delete c1;

}




void DrawPlot(TH1D* h1, bool logy, bool fit, string x_title, string y_axis, string label, string out_file, int ptKick)
{

  TCanvas* c1 = new TCanvas(x_title.c_str(),"",550,500);
  c1->cd();

  h1->SetLineColor(kAzure+1);
  h1->SetFillColorAlpha(kAzure,0.3);
  if( x_title != "keep") h1->GetXaxis()->SetTitle(x_title.c_str());
  h1->GetXaxis()->SetTitleSize(0.05);
  h1->GetXaxis()->SetTitleOffset(1.25);
  h1->GetYaxis()->SetTitleOffset(1.45);
  if( y_axis != "keep") h1->GetYaxis()->SetTitle(y_axis.c_str());
  h1->GetYaxis()->SetTitleSize(0.05);
   gStyle->SetOptStat("rme");
  if( fit ){
    gStyle->SetOptStat("e");
    GaussianFit( h1 );
  }
  h1->Draw();

  if(logy){
    gPad->SetLogy();
     h1->GetYaxis()->SetRangeUser(1e-6, h1->GetMaximum()*100.);
  }
  /*if(h1.size() > 1){
    for(int i = 1; i < h1.size(); i++){
      h1.at(i)->Draw("SAME");
      leg->AddEntry(h1.at(i),h1.at(i)->GetTitle(),"l");
    }
  }*/
  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.05);
  gPad->SetLeftMargin(0.18);
  gPad->SetBottomMargin(0.13);

  TLatex *lat = new TLatex();
  lat->SetTextFont(72);
  lat->SetTextSize(0.04);
  lat->SetTextFont(42);
  lat->DrawLatexNDC(.2,.9,label.c_str());
  //leg->SetTextSize(0.035  );
  //leg->Draw();
  gPad->Update();
  /*
  TPaveStats *st = (TPaveStats*)h1->FindObject("stats");
  st->SetX1NDC(0.5);
  st->SetX2NDC(0.95);
  st->SetY1NDC(0.85);
  st->SetY2NDC(0.99);
  */
  gPad->Update();
  c1->Print(out_file.c_str());
  delete c1;

}


void DrawChargeMap(TH2D* h1, string label, string out_file)
{

  TCanvas* c1 = new TCanvas("c1","",600,500);
  c1->cd();
  h1->GetXaxis()->SetTitle("X [cm]");
  h1->GetYaxis()->SetTitle("Y [cm]");
  h1->GetZaxis()->SetTitle("Charge [a.u.]");
  //  h1->GetZaxis()->SetTitle("Charge");
  h1->SetContour(99);
  h1->Scale(1./h1->Integral());

  h1->GetXaxis()->SetTitleSize(0.05);
  h1->GetXaxis()->SetTitleOffset(1.25);
  h1->GetYaxis()->SetTitleOffset(1.25);
  h1->GetYaxis()->SetTitleSize(0.05);
  h1->GetZaxis()->SetTitleOffset(1.35);
  h1->Draw("colz");

  //  gPad->SetLogz();
  // by default: left (0.15), right (0.15), top (0.05), bottom (0.15)
  //  gPad->SetTopMargin(0.05);
  gPad->SetRightMargin(0.18);
  //  gPad->SetLeftMargin(0.15);
  //  gPad->SetBottomMargin(0.15);
  //  std::cout << " left marg " << gPad->GetLeftMargin() << "right marg " << gPad->GetRightMargin() << " top marg " << gPad->GetTopMargin() << " bottom marg " << gPad->GetBottomMargin() << std::endl;
  TLatex *lat = new TLatex();
  lat->SetTextFont(72);
  lat->SetTextSize(0.04);
  lat->SetTextFont(42);
  lat->SetTextColor(0);
  lat->DrawLatexNDC(.17,.82,label.c_str());
  //leg->SetTextSize(0.035  );
  //leg->Draw();
  gPad->Update();
  /*
  TPaveStats *st = (TPaveStats*)h1->FindObject("stats");
  st->SetX1NDC(0.5);
  st->SetX2NDC(0.95);
  st->SetY1NDC(0.85);
  st->SetY2NDC(0.99);
  */
  gPad->Update();
  c1->Print(out_file.c_str());
  delete c1;

}

void FindCOM(vector<double> rpdSignals, double &comX, double &comY) {

  double totalSignal = 0;
  comX = 0; comY = 0;
  for (int ch = 0; ch < rpdSignals.size(); ++ch) {
    float x; float y;
    if (ch < 4) y = 1.5;
    else if (ch >= 4 && ch < 8) y = 0.5;
    else if (ch >= 8 && ch < 12) y = -0.5;
    else y = -1.5;
    if (ch % 4 == 0) x = 1.5;
    else if (ch % 4 == 1) x = 0.5;
    else if (ch % 4 == 2) x = -0.5;
    else if (ch % 4 == 3) x = -1.5;
    totalSignal += rpdSignals[ch];
    comX += (x * rpdSignals[ch]);
    comY += (y * rpdSignals[ch]);
  }
  comX /= totalSignal; 
  comY /= totalSignal; 
 
}

double GetCOMReactionPlane(double comX, double comY, double centerX, double centerY) {  
  double opp = comY - centerY;
  double adj = comX - centerX;
  double phi = TMath::ATan2(opp,adj);
  return phi;
}
