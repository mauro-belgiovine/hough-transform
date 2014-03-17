//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Dec 16 10:13:51 2013 by ROOT version 5.34/13
// from TTree CombinedInDetTracks/CombinedInDetTracks Validation
// found on file: /tmp/sidoti/mc10_7TeV.105001.pythia_minbias.recon.NTUP_TRKVALID.e574_s932_s946_r2507_tid458842_00/NTUP_TRKVALID.458842._000824.root.1
//////////////////////////////////////////////////////////

#ifndef trkvalid_h
#define trkvalid_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include <vector>


// start AS
#include <iostream>
#include <fstream>
#include <stdio.h>
using namespace std;
// end AS

// Fixed size dimensions of array or collections stored in the TTree if any.

class trkvalid {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   Int_t           RunNumber;
   Int_t           EventNumber;
   Int_t           TrackID;
   Int_t           IterationIndex;
   Int_t           nHits;
   Float_t         Chi2overNdof;
   Int_t           Ndof;
   Int_t           FitterStatusCode;
   Int_t           TrackFitAuthor;
   Int_t           TrackSeedAuthor;
   Int_t           TrackParticleHypo;
   Float_t         RecD0;
   Float_t         RecZ0;
   Float_t         RecPhi0;
   Float_t         RecTheta;
   Float_t         RecEta;
   Float_t         RecQoverP;
   Float_t         RecErrD0;
   Float_t         RecErrZ0;
   Float_t         RecErrPhi0;
   Float_t         RecErrTheta;
   Float_t         RecErrQoverP;
   Float_t         trk_Mc_d0;
   Float_t         trk_Mc_z0;
   Float_t         trk_Mc_phi0;
   Float_t         trk_Mc_theta;
   Float_t         trk_Mc_qOverP;
   Float_t         trk_Mc_qOverPt;
   Float_t         trk_Mc_eta;
   Float_t         trk_Mc_diff_d0;
   Float_t         trk_Mc_diff_z0;
   Float_t         trk_Mc_diff_phi0;
   Float_t         trk_Mc_diff_theta;
   Float_t         trk_Mc_diff_qOverP;
   Float_t         trk_Mc_pull_d0;
   Float_t         trk_Mc_pull_z0;
   Float_t         trk_Mc_pull_phi0;
   Float_t         trk_Mc_pull_theta;
   Float_t         trk_Mc_pull_qOverP;
   Int_t           trk_Mc_particleID;
   Int_t           trk_Mc_barcode;
   Float_t         trk_Mc_energy;
   Float_t         trk_Mc_prob;
   Int_t           trk_Mc_truthTreeIndex;
   Int_t           TrackStatesUnbiased;
   vector<float>   *pullLocX;
   vector<float>   *pullLocY;
   vector<float>   *residualLocX;
   vector<float>   *residualLocY;
   vector<int>     *DetectorType;
   vector<int>     *outlierFlag;
   Int_t           nPixelHits;
   Int_t           nSCTHits;
   Int_t           nTRTHits;
   Int_t           nMDTHits;
   Int_t           nCSCHits;
   Int_t           nRPCHits;
   Int_t           nTGCHits;
   vector<int>     *pixelHitIndex;
   vector<int>     *sctHitIndex;
   vector<int>     *trtHitIndex;
   vector<int>     *mdtHitIndex;
   vector<int>     *cscHitIndex;
   vector<int>     *rpcHitIndex;
   vector<int>     *tgcHitIndex;
   vector<float>   *PixPullPhi;
   vector<float>   *PixPullEta;
   vector<float>   *PixResPhi;
   vector<float>   *PixResEta;
   vector<int>     *PixDetType;
   vector<int>     *PixHasGanged;
   vector<int>     *PixIsFake;
   vector<int>     *PixLVL1A;
   vector<int>     *PixToT;
   vector<float>   *PixCharge;
   vector<float>   *PixTLorPhi;
   vector<float>   *PixTLorEta;
   vector<float>   *PixBiasVolt;
   vector<float>   *PixTemp;
   vector<float>   *PixDepVolt;
   vector<float>   *PixDigResPhi;
   vector<float>   *PixDigResEta;
   vector<float>   *PixErrPhi;
   vector<float>   *PixErrEta;
   vector<float>   *PixLocX;
   vector<float>   *PixLocY;
   vector<int>     *PixEtaModule;
   vector<int>     *PixPhiModule;
   vector<float>   *PixGloX;
   vector<float>   *PixGloY;
   vector<float>   *PixGloZ;
   vector<float>   *PixEta;
   vector<float>   *PixFirstRow;
   vector<float>   *PixFirstCol;
   vector<int>     *PixDeltaRow;
   vector<int>     *PixDeltaCol;
   vector<float>   *PixDeltaPhi;
   vector<float>   *PixDeltaEta;
   vector<float>   *PixOmegaPhi;
   vector<float>   *PixOmegaEta;
   vector<float>   *PixTrkAngle;
   vector<float>   *PixTrkThetaI;
   vector<float>   *PixTrkEta;
   vector<float>   *PixTrkPt;
   vector<float>   *PixTrkQ;
   vector<float>   *PixTrkClusGroupSize;
   vector<float>   *PixHoleGloX;
   vector<float>   *PixHoleGloY;
   vector<float>   *PixHoleGloZ;
   vector<float>   *PixHoleLocX;
   vector<float>   *PixHoleLocY;
   vector<int>     *PixHoleEtaModule;
   vector<int>     *PixHolePhiModule;
   vector<float>   *PixHoleLayer;
   vector<float>   *PixHoleECBarrel;
   vector<int>     *PixHoleClNearstSize;
   vector<float>   *PixHoleLocXClNearst;
   vector<float>   *PixHoleLocYClNearst;
   vector<float>   *PixHoleClRowClNearst;
   vector<float>   *PixHoleClColClNearst;
   vector<float>   *PixHoleGloZNearst;
   vector<float>   *PixHoleDeltaRowNearst;
   vector<float>   *PixHoleDeltaColNearst;
   Int_t           LVL1TriggerType;
   vector<float>   *SCTPull;
   vector<float>   *SCTRes;
   vector<float>   *TRTPull;
   vector<float>   *TRTRes;
   vector<float>   *HitX;
   vector<float>   *HitY;
   vector<float>   *HitZ;
   vector<float>   *HitRadius;
   vector<float>   *HitPhi;
   vector<float>   *HitLocal1;
   vector<float>   *HitLocal2;
   vector<float>   *HitSurfaceX;
   vector<float>   *HitSurfaceY;
   vector<float>   *HitSurfaceZ;
   vector<float>   *HitSurfaceRadius;
   vector<float>   *HitSurfacePhi;
   vector<float>   *HitSurfaceTheta;
   vector<float>   *TrackX;
   vector<float>   *TrackY;
   vector<float>   *TrackZ;
   vector<float>   *TrackMomentumX;
   vector<float>   *TrackMomentumY;
   vector<float>   *TrackMomentumZ;
   vector<float>   *TrackLocal1;
   vector<float>   *TrackLocal2;

   // List of branches
   TBranch        *b_RunNumber;   //!
   TBranch        *b_EventNumber;   //!
   TBranch        *b_TrackID;   //!
   TBranch        *b_IterationIndex;   //!
   TBranch        *b_nHits;   //!
   TBranch        *b_Chi2overNdof;   //!
   TBranch        *b_Ndof;   //!
   TBranch        *b_FitterStatusCode;   //!
   TBranch        *b_TrackFitAuthor;   //!
   TBranch        *b_TrackSeedAuthor;   //!
   TBranch        *b_TrackParticleHypo;   //!
   TBranch        *b_RecD0;   //!
   TBranch        *b_RecZ0;   //!
   TBranch        *b_RecPhi0;   //!
   TBranch        *b_RecTheta;   //!
   TBranch        *b_RecEta;   //!
   TBranch        *b_RecQoverP;   //!
   TBranch        *b_RecErrD0;   //!
   TBranch        *b_RecErrZ0;   //!
   TBranch        *b_RecErrPhi0;   //!
   TBranch        *b_RecErrTheta;   //!
   TBranch        *b_RecErrQoverP;   //!
   TBranch        *b_trk_Mc_d0;   //!
   TBranch        *b_trk_Mc_z0;   //!
   TBranch        *b_trk_Mc_phi0;   //!
   TBranch        *b_trk_Mc_theta;   //!
   TBranch        *b_trk_Mc_qOverP;   //!
   TBranch        *b_trk_Mc_qOverPt;   //!
   TBranch        *b_trk_Mc_eta;   //!
   TBranch        *b_trk_Mc_diff_d0;   //!
   TBranch        *b_trk_Mc_diff_z0;   //!
   TBranch        *b_trk_Mc_diff_phi0;   //!
   TBranch        *b_trk_Mc_diff_theta;   //!
   TBranch        *b_trk_Mc_diff_qOverP;   //!
   TBranch        *b_trk_Mc_pull_d0;   //!
   TBranch        *b_trk_Mc_pull_z0;   //!
   TBranch        *b_trk_Mc_pull_phi0;   //!
   TBranch        *b_trk_Mc_pull_theta;   //!
   TBranch        *b_trk_Mc_pull_qOverP;   //!
   TBranch        *b_trk_Mc_particleID;   //!
   TBranch        *b_trk_Mc_barcode;   //!
   TBranch        *b_trk_Mc_energy;   //!
   TBranch        *b_trk_Mc_prob;   //!
   TBranch        *b_trk_Mc_truthTreeIndex;   //!
   TBranch        *b_TrackStatesUnbiased;   //!
   TBranch        *b_pullLocX;   //!
   TBranch        *b_pullLocY;   //!
   TBranch        *b_residualLocX;   //!
   TBranch        *b_residualLocY;   //!
   TBranch        *b_DetectorType;   //!
   TBranch        *b_outlierFlag;   //!
   TBranch        *b_nPixelHits;   //!
   TBranch        *b_nSCTHits;   //!
   TBranch        *b_nTRTHits;   //!
   TBranch        *b_nMDTHits;   //!
   TBranch        *b_nCSCHits;   //!
   TBranch        *b_nRPCHits;   //!
   TBranch        *b_nTGCHits;   //!
   TBranch        *b_pixelHitIndex;   //!
   TBranch        *b_sctHitIndex;   //!
   TBranch        *b_trtHitIndex;   //!
   TBranch        *b_mdtHitIndex;   //!
   TBranch        *b_cscHitIndex;   //!
   TBranch        *b_rpcHitIndex;   //!
   TBranch        *b_tgcHitIndex;   //!
   TBranch        *b_PixPullPhi;   //!
   TBranch        *b_PixPullEta;   //!
   TBranch        *b_PixResPhi;   //!
   TBranch        *b_PixResEta;   //!
   TBranch        *b_PixDetType;   //!
   TBranch        *b_PixHasGanged;   //!
   TBranch        *b_PixIsFake;   //!
   TBranch        *b_PixLVL1A;   //!
   TBranch        *b_PixToT;   //!
   TBranch        *b_PixCharge;   //!
   TBranch        *b_PixTLorPhi;   //!
   TBranch        *b_PixTLorEta;   //!
   TBranch        *b_PixBiasVolt;   //!
   TBranch        *b_PixTemp;   //!
   TBranch        *b_PixDepVolt;   //!
   TBranch        *b_PixDigResPhi;   //!
   TBranch        *b_PixDigResEta;   //!
   TBranch        *b_PixErrPhi;   //!
   TBranch        *b_PixErrEta;   //!
   TBranch        *b_PixLocX;   //!
   TBranch        *b_PixLocY;   //!
   TBranch        *b_PixEtaModule;   //!
   TBranch        *b_PixPhiModule;   //!
   TBranch        *b_PixGloX;   //!
   TBranch        *b_PixGloY;   //!
   TBranch        *b_PixGloZ;   //!
   TBranch        *b_PixEta;   //!
   TBranch        *b_PixFirstRow;   //!
   TBranch        *b_PixFirstCol;   //!
   TBranch        *b_PixDeltaRow;   //!
   TBranch        *b_PixDeltaCol;   //!
   TBranch        *b_PixDeltaPhi;   //!
   TBranch        *b_PixDeltaEta;   //!
   TBranch        *b_PixOmegaPhi;   //!
   TBranch        *b_PixOmegaEta;   //!
   TBranch        *b_PixTrkAngle;   //!
   TBranch        *b_PixTrkThetaI;   //!
   TBranch        *b_PixTrkEta;   //!
   TBranch        *b_PixTrkPt;   //!
   TBranch        *b_PixTrkQ;   //!
   TBranch        *b_PixTrkClusGroupSize;   //!
   TBranch        *b_PixHoleGloX;   //!
   TBranch        *b_PixHoleGloY;   //!
   TBranch        *b_PixHoleGloZ;   //!
   TBranch        *b_PixHoleLocX;   //!
   TBranch        *b_PixHoleLocY;   //!
   TBranch        *b_PixHoleEtaModule;   //!
   TBranch        *b_PixHolePhiModule;   //!
   TBranch        *b_PixHoleLayer;   //!
   TBranch        *b_PixHoleECBarrel;   //!
   TBranch        *b_PixHoleClNearstSize;   //!
   TBranch        *b_PixHoleLocXClNearst;   //!
   TBranch        *b_PixHoleLocYClNearst;   //!
   TBranch        *b_PixHoleClRowClNearst;   //!
   TBranch        *b_PixHoleClColClNearst;   //!
   TBranch        *b_PixHoleGloZNearst;   //!
   TBranch        *b_PixHoleDeltaRowNearst;   //!
   TBranch        *b_PixHoleDeltaColNearst;   //!
   TBranch        *b_LVL1TriggerType;   //!
   TBranch        *b_SCTPull;   //!
   TBranch        *b_SCTRes;   //!
   TBranch        *b_TRTPull;   //!
   TBranch        *b_TRTRes;   //!
   TBranch        *b_HitX;   //!
   TBranch        *b_HitY;   //!
   TBranch        *b_HitZ;   //!
   TBranch        *b_HitRadius;   //!
   TBranch        *b_HitPhi;   //!
   TBranch        *b_HitLocal1;   //!
   TBranch        *b_HitLocal2;   //!
   TBranch        *b_HitSurfaceX;   //!
   TBranch        *b_HitSurfaceY;   //!
   TBranch        *b_HitSurfaceZ;   //!
   TBranch        *b_HitSurfaceRadius;   //!
   TBranch        *b_HitSurfacePhi;   //!
   TBranch        *b_HitSurfaceTheta;   //!
   TBranch        *b_TrackX;   //!
   TBranch        *b_TrackY;   //!
   TBranch        *b_TrackZ;   //!
   TBranch        *b_TrackMomentumX;   //!
   TBranch        *b_TrackMomentumY;   //!
   TBranch        *b_TrackMomentumZ;   //!
   TBranch        *b_TrackLocal1;   //!
   TBranch        *b_TrackLocal2;   //!

   trkvalid(TTree *tree=0);
   virtual ~trkvalid();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
  // start AS
  string m_output_file;
  inline void SetOutputFile(string m_string){
    m_output_file = m_string;
  }
  
  // end AS

};

#endif

#ifdef trkvalid_cxx
trkvalid::trkvalid(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/tmp/sidoti/mc10_7TeV.105001.pythia_minbias.recon.NTUP_TRKVALID.e574_s932_s946_r2507_tid458842_00/NTUP_TRKVALID.458842._000824.root.1");
      if (!f || !f->IsOpen()) {
         f = new TFile("/tmp/sidoti/mc10_7TeV.105001.pythia_minbias.recon.NTUP_TRKVALID.e574_s932_s946_r2507_tid458842_00/NTUP_TRKVALID.458842._000824.root.1");
      }
      TDirectory * dir = (TDirectory*)f->Get("/tmp/sidoti/mc10_7TeV.105001.pythia_minbias.recon.NTUP_TRKVALID.e574_s932_s946_r2507_tid458842_00/NTUP_TRKVALID.458842._000824.root.1:/Validation");
      dir->GetObject("CombinedInDetTracks",tree);

   }
   Init(tree);
}

trkvalid::~trkvalid()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t trkvalid::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t trkvalid::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void trkvalid::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   pullLocX = 0;
   pullLocY = 0;
   residualLocX = 0;
   residualLocY = 0;
   DetectorType = 0;
   outlierFlag = 0;
   pixelHitIndex = 0;
   sctHitIndex = 0;
   trtHitIndex = 0;
   mdtHitIndex = 0;
   cscHitIndex = 0;
   rpcHitIndex = 0;
   tgcHitIndex = 0;
   PixPullPhi = 0;
   PixPullEta = 0;
   PixResPhi = 0;
   PixResEta = 0;
   PixDetType = 0;
   PixHasGanged = 0;
   PixIsFake = 0;
   PixLVL1A = 0;
   PixToT = 0;
   PixCharge = 0;
   PixTLorPhi = 0;
   PixTLorEta = 0;
   PixBiasVolt = 0;
   PixTemp = 0;
   PixDepVolt = 0;
   PixDigResPhi = 0;
   PixDigResEta = 0;
   PixErrPhi = 0;
   PixErrEta = 0;
   PixLocX = 0;
   PixLocY = 0;
   PixEtaModule = 0;
   PixPhiModule = 0;
   PixGloX = 0;
   PixGloY = 0;
   PixGloZ = 0;
   PixEta = 0;
   PixFirstRow = 0;
   PixFirstCol = 0;
   PixDeltaRow = 0;
   PixDeltaCol = 0;
   PixDeltaPhi = 0;
   PixDeltaEta = 0;
   PixOmegaPhi = 0;
   PixOmegaEta = 0;
   PixTrkAngle = 0;
   PixTrkThetaI = 0;
   PixTrkEta = 0;
   PixTrkPt = 0;
   PixTrkQ = 0;
   PixTrkClusGroupSize = 0;
   PixHoleGloX = 0;
   PixHoleGloY = 0;
   PixHoleGloZ = 0;
   PixHoleLocX = 0;
   PixHoleLocY = 0;
   PixHoleEtaModule = 0;
   PixHolePhiModule = 0;
   PixHoleLayer = 0;
   PixHoleECBarrel = 0;
   PixHoleClNearstSize = 0;
   PixHoleLocXClNearst = 0;
   PixHoleLocYClNearst = 0;
   PixHoleClRowClNearst = 0;
   PixHoleClColClNearst = 0;
   PixHoleGloZNearst = 0;
   PixHoleDeltaRowNearst = 0;
   PixHoleDeltaColNearst = 0;
   SCTPull = 0;
   SCTRes = 0;
   TRTPull = 0;
   TRTRes = 0;
   HitX = 0;
   HitY = 0;
   HitZ = 0;
   HitRadius = 0;
   HitPhi = 0;
   HitLocal1 = 0;
   HitLocal2 = 0;
   HitSurfaceX = 0;
   HitSurfaceY = 0;
   HitSurfaceZ = 0;
   HitSurfaceRadius = 0;
   HitSurfacePhi = 0;
   HitSurfaceTheta = 0;
   TrackX = 0;
   TrackY = 0;
   TrackZ = 0;
   TrackMomentumX = 0;
   TrackMomentumY = 0;
   TrackMomentumZ = 0;
   TrackLocal1 = 0;
   TrackLocal2 = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("RunNumber", &RunNumber, &b_RunNumber);
   fChain->SetBranchAddress("EventNumber", &EventNumber, &b_EventNumber);
   fChain->SetBranchAddress("TrackID", &TrackID, &b_TrackID);
   fChain->SetBranchAddress("IterationIndex", &IterationIndex, &b_IterationIndex);
   fChain->SetBranchAddress("nHits", &nHits, &b_nHits);
   fChain->SetBranchAddress("Chi2overNdof", &Chi2overNdof, &b_Chi2overNdof);
   fChain->SetBranchAddress("Ndof", &Ndof, &b_Ndof);
   fChain->SetBranchAddress("FitterStatusCode", &FitterStatusCode, &b_FitterStatusCode);
   fChain->SetBranchAddress("TrackFitAuthor", &TrackFitAuthor, &b_TrackFitAuthor);
   fChain->SetBranchAddress("TrackSeedAuthor", &TrackSeedAuthor, &b_TrackSeedAuthor);
   fChain->SetBranchAddress("TrackParticleHypo", &TrackParticleHypo, &b_TrackParticleHypo);
   fChain->SetBranchAddress("RecD0", &RecD0, &b_RecD0);
   fChain->SetBranchAddress("RecZ0", &RecZ0, &b_RecZ0);
   fChain->SetBranchAddress("RecPhi0", &RecPhi0, &b_RecPhi0);
   fChain->SetBranchAddress("RecTheta", &RecTheta, &b_RecTheta);
   fChain->SetBranchAddress("RecEta", &RecEta, &b_RecEta);
   fChain->SetBranchAddress("RecQoverP", &RecQoverP, &b_RecQoverP);
   fChain->SetBranchAddress("RecErrD0", &RecErrD0, &b_RecErrD0);
   fChain->SetBranchAddress("RecErrZ0", &RecErrZ0, &b_RecErrZ0);
   fChain->SetBranchAddress("RecErrPhi0", &RecErrPhi0, &b_RecErrPhi0);
   fChain->SetBranchAddress("RecErrTheta", &RecErrTheta, &b_RecErrTheta);
   fChain->SetBranchAddress("RecErrQoverP", &RecErrQoverP, &b_RecErrQoverP);
   fChain->SetBranchAddress("trk_Mc_d0", &trk_Mc_d0, &b_trk_Mc_d0);
   fChain->SetBranchAddress("trk_Mc_z0", &trk_Mc_z0, &b_trk_Mc_z0);
   fChain->SetBranchAddress("trk_Mc_phi0", &trk_Mc_phi0, &b_trk_Mc_phi0);
   fChain->SetBranchAddress("trk_Mc_theta", &trk_Mc_theta, &b_trk_Mc_theta);
   fChain->SetBranchAddress("trk_Mc_qOverP", &trk_Mc_qOverP, &b_trk_Mc_qOverP);
   fChain->SetBranchAddress("trk_Mc_qOverPt", &trk_Mc_qOverPt, &b_trk_Mc_qOverPt);
   fChain->SetBranchAddress("trk_Mc_eta", &trk_Mc_eta, &b_trk_Mc_eta);
   fChain->SetBranchAddress("trk_Mc_diff_d0", &trk_Mc_diff_d0, &b_trk_Mc_diff_d0);
   fChain->SetBranchAddress("trk_Mc_diff_z0", &trk_Mc_diff_z0, &b_trk_Mc_diff_z0);
   fChain->SetBranchAddress("trk_Mc_diff_phi0", &trk_Mc_diff_phi0, &b_trk_Mc_diff_phi0);
   fChain->SetBranchAddress("trk_Mc_diff_theta", &trk_Mc_diff_theta, &b_trk_Mc_diff_theta);
   fChain->SetBranchAddress("trk_Mc_diff_qOverP", &trk_Mc_diff_qOverP, &b_trk_Mc_diff_qOverP);
   fChain->SetBranchAddress("trk_Mc_pull_d0", &trk_Mc_pull_d0, &b_trk_Mc_pull_d0);
   fChain->SetBranchAddress("trk_Mc_pull_z0", &trk_Mc_pull_z0, &b_trk_Mc_pull_z0);
   fChain->SetBranchAddress("trk_Mc_pull_phi0", &trk_Mc_pull_phi0, &b_trk_Mc_pull_phi0);
   fChain->SetBranchAddress("trk_Mc_pull_theta", &trk_Mc_pull_theta, &b_trk_Mc_pull_theta);
   fChain->SetBranchAddress("trk_Mc_pull_qOverP", &trk_Mc_pull_qOverP, &b_trk_Mc_pull_qOverP);
   fChain->SetBranchAddress("trk_Mc_particleID", &trk_Mc_particleID, &b_trk_Mc_particleID);
   fChain->SetBranchAddress("trk_Mc_barcode", &trk_Mc_barcode, &b_trk_Mc_barcode);
   fChain->SetBranchAddress("trk_Mc_energy", &trk_Mc_energy, &b_trk_Mc_energy);
   fChain->SetBranchAddress("trk_Mc_prob", &trk_Mc_prob, &b_trk_Mc_prob);
   fChain->SetBranchAddress("trk_Mc_truthTreeIndex", &trk_Mc_truthTreeIndex, &b_trk_Mc_truthTreeIndex);
   fChain->SetBranchAddress("TrackStatesUnbiased", &TrackStatesUnbiased, &b_TrackStatesUnbiased);
   fChain->SetBranchAddress("pullLocX", &pullLocX, &b_pullLocX);
   fChain->SetBranchAddress("pullLocY", &pullLocY, &b_pullLocY);
   fChain->SetBranchAddress("residualLocX", &residualLocX, &b_residualLocX);
   fChain->SetBranchAddress("residualLocY", &residualLocY, &b_residualLocY);
   fChain->SetBranchAddress("DetectorType", &DetectorType, &b_DetectorType);
   fChain->SetBranchAddress("outlierFlag", &outlierFlag, &b_outlierFlag);
   fChain->SetBranchAddress("nPixelHits", &nPixelHits, &b_nPixelHits);
   fChain->SetBranchAddress("nSCTHits", &nSCTHits, &b_nSCTHits);
   fChain->SetBranchAddress("nTRTHits", &nTRTHits, &b_nTRTHits);
   fChain->SetBranchAddress("nMDTHits", &nMDTHits, &b_nMDTHits);
   fChain->SetBranchAddress("nCSCHits", &nCSCHits, &b_nCSCHits);
   fChain->SetBranchAddress("nRPCHits", &nRPCHits, &b_nRPCHits);
   fChain->SetBranchAddress("nTGCHits", &nTGCHits, &b_nTGCHits);
   fChain->SetBranchAddress("pixelHitIndex", &pixelHitIndex, &b_pixelHitIndex);
   fChain->SetBranchAddress("sctHitIndex", &sctHitIndex, &b_sctHitIndex);
   fChain->SetBranchAddress("trtHitIndex", &trtHitIndex, &b_trtHitIndex);
   fChain->SetBranchAddress("mdtHitIndex", &mdtHitIndex, &b_mdtHitIndex);
   fChain->SetBranchAddress("cscHitIndex", &cscHitIndex, &b_cscHitIndex);
   fChain->SetBranchAddress("rpcHitIndex", &rpcHitIndex, &b_rpcHitIndex);
   fChain->SetBranchAddress("tgcHitIndex", &tgcHitIndex, &b_tgcHitIndex);
   fChain->SetBranchAddress("PixPullPhi", &PixPullPhi, &b_PixPullPhi);
   fChain->SetBranchAddress("PixPullEta", &PixPullEta, &b_PixPullEta);
   fChain->SetBranchAddress("PixResPhi", &PixResPhi, &b_PixResPhi);
   fChain->SetBranchAddress("PixResEta", &PixResEta, &b_PixResEta);
   fChain->SetBranchAddress("PixDetType", &PixDetType, &b_PixDetType);
   fChain->SetBranchAddress("PixHasGanged", &PixHasGanged, &b_PixHasGanged);
   fChain->SetBranchAddress("PixIsFake", &PixIsFake, &b_PixIsFake);
   fChain->SetBranchAddress("PixLVL1A", &PixLVL1A, &b_PixLVL1A);
   fChain->SetBranchAddress("PixToT", &PixToT, &b_PixToT);
   fChain->SetBranchAddress("PixCharge", &PixCharge, &b_PixCharge);
   fChain->SetBranchAddress("PixTLorPhi", &PixTLorPhi, &b_PixTLorPhi);
   fChain->SetBranchAddress("PixTLorEta", &PixTLorEta, &b_PixTLorEta);
   fChain->SetBranchAddress("PixBiasVolt", &PixBiasVolt, &b_PixBiasVolt);
   fChain->SetBranchAddress("PixTemp", &PixTemp, &b_PixTemp);
   fChain->SetBranchAddress("PixDepVolt", &PixDepVolt, &b_PixDepVolt);
   fChain->SetBranchAddress("PixDigResPhi", &PixDigResPhi, &b_PixDigResPhi);
   fChain->SetBranchAddress("PixDigResEta", &PixDigResEta, &b_PixDigResEta);
   fChain->SetBranchAddress("PixErrPhi", &PixErrPhi, &b_PixErrPhi);
   fChain->SetBranchAddress("PixErrEta", &PixErrEta, &b_PixErrEta);
   fChain->SetBranchAddress("PixLocX", &PixLocX, &b_PixLocX);
   fChain->SetBranchAddress("PixLocY", &PixLocY, &b_PixLocY);
   fChain->SetBranchAddress("PixEtaModule", &PixEtaModule, &b_PixEtaModule);
   fChain->SetBranchAddress("PixPhiModule", &PixPhiModule, &b_PixPhiModule);
   fChain->SetBranchAddress("PixGloX", &PixGloX, &b_PixGloX);
   fChain->SetBranchAddress("PixGloY", &PixGloY, &b_PixGloY);
   fChain->SetBranchAddress("PixGloZ", &PixGloZ, &b_PixGloZ);
   fChain->SetBranchAddress("PixEta", &PixEta, &b_PixEta);
   fChain->SetBranchAddress("PixFirstRow", &PixFirstRow, &b_PixFirstRow);
   fChain->SetBranchAddress("PixFirstCol", &PixFirstCol, &b_PixFirstCol);
   fChain->SetBranchAddress("PixDeltaRow", &PixDeltaRow, &b_PixDeltaRow);
   fChain->SetBranchAddress("PixDeltaCol", &PixDeltaCol, &b_PixDeltaCol);
   fChain->SetBranchAddress("PixDeltaPhi", &PixDeltaPhi, &b_PixDeltaPhi);
   fChain->SetBranchAddress("PixDeltaEta", &PixDeltaEta, &b_PixDeltaEta);
   fChain->SetBranchAddress("PixOmegaPhi", &PixOmegaPhi, &b_PixOmegaPhi);
   fChain->SetBranchAddress("PixOmegaEta", &PixOmegaEta, &b_PixOmegaEta);
   fChain->SetBranchAddress("PixTrkAngle", &PixTrkAngle, &b_PixTrkAngle);
   fChain->SetBranchAddress("PixTrkThetaI", &PixTrkThetaI, &b_PixTrkThetaI);
   fChain->SetBranchAddress("PixTrkEta", &PixTrkEta, &b_PixTrkEta);
   fChain->SetBranchAddress("PixTrkPt", &PixTrkPt, &b_PixTrkPt);
   fChain->SetBranchAddress("PixTrkQ", &PixTrkQ, &b_PixTrkQ);
   fChain->SetBranchAddress("PixTrkClusGroupSize", &PixTrkClusGroupSize, &b_PixTrkClusGroupSize);
   fChain->SetBranchAddress("PixHoleGloX", &PixHoleGloX, &b_PixHoleGloX);
   fChain->SetBranchAddress("PixHoleGloY", &PixHoleGloY, &b_PixHoleGloY);
   fChain->SetBranchAddress("PixHoleGloZ", &PixHoleGloZ, &b_PixHoleGloZ);
   fChain->SetBranchAddress("PixHoleLocX", &PixHoleLocX, &b_PixHoleLocX);
   fChain->SetBranchAddress("PixHoleLocY", &PixHoleLocY, &b_PixHoleLocY);
   fChain->SetBranchAddress("PixHoleEtaModule", &PixHoleEtaModule, &b_PixHoleEtaModule);
   fChain->SetBranchAddress("PixHolePhiModule", &PixHolePhiModule, &b_PixHolePhiModule);
   fChain->SetBranchAddress("PixHoleLayer", &PixHoleLayer, &b_PixHoleLayer);
   fChain->SetBranchAddress("PixHoleECBarrel", &PixHoleECBarrel, &b_PixHoleECBarrel);
   fChain->SetBranchAddress("PixHoleClNearstSize", &PixHoleClNearstSize, &b_PixHoleClNearstSize);
   fChain->SetBranchAddress("PixHoleLocXClNearst", &PixHoleLocXClNearst, &b_PixHoleLocXClNearst);
   fChain->SetBranchAddress("PixHoleLocYClNearst", &PixHoleLocYClNearst, &b_PixHoleLocYClNearst);
   fChain->SetBranchAddress("PixHoleClRowClNearst", &PixHoleClRowClNearst, &b_PixHoleClRowClNearst);
   fChain->SetBranchAddress("PixHoleClColClNearst", &PixHoleClColClNearst, &b_PixHoleClColClNearst);
   fChain->SetBranchAddress("PixHoleGloZNearst", &PixHoleGloZNearst, &b_PixHoleGloZNearst);
   fChain->SetBranchAddress("PixHoleDeltaRowNearst", &PixHoleDeltaRowNearst, &b_PixHoleDeltaRowNearst);
   fChain->SetBranchAddress("PixHoleDeltaColNearst", &PixHoleDeltaColNearst, &b_PixHoleDeltaColNearst);
   fChain->SetBranchAddress("LVL1TriggerType", &LVL1TriggerType, &b_LVL1TriggerType);
   fChain->SetBranchAddress("SCTPull", &SCTPull, &b_SCTPull);
   fChain->SetBranchAddress("SCTRes", &SCTRes, &b_SCTRes);
   fChain->SetBranchAddress("TRTPull", &TRTPull, &b_TRTPull);
   fChain->SetBranchAddress("TRTRes", &TRTRes, &b_TRTRes);
   fChain->SetBranchAddress("HitX", &HitX, &b_HitX);
   fChain->SetBranchAddress("HitY", &HitY, &b_HitY);
   fChain->SetBranchAddress("HitZ", &HitZ, &b_HitZ);
   fChain->SetBranchAddress("HitRadius", &HitRadius, &b_HitRadius);
   fChain->SetBranchAddress("HitPhi", &HitPhi, &b_HitPhi);
   fChain->SetBranchAddress("HitLocal1", &HitLocal1, &b_HitLocal1);
   fChain->SetBranchAddress("HitLocal2", &HitLocal2, &b_HitLocal2);
   fChain->SetBranchAddress("HitSurfaceX", &HitSurfaceX, &b_HitSurfaceX);
   fChain->SetBranchAddress("HitSurfaceY", &HitSurfaceY, &b_HitSurfaceY);
   fChain->SetBranchAddress("HitSurfaceZ", &HitSurfaceZ, &b_HitSurfaceZ);
   fChain->SetBranchAddress("HitSurfaceRadius", &HitSurfaceRadius, &b_HitSurfaceRadius);
   fChain->SetBranchAddress("HitSurfacePhi", &HitSurfacePhi, &b_HitSurfacePhi);
   fChain->SetBranchAddress("HitSurfaceTheta", &HitSurfaceTheta, &b_HitSurfaceTheta);
   fChain->SetBranchAddress("TrackX", &TrackX, &b_TrackX);
   fChain->SetBranchAddress("TrackY", &TrackY, &b_TrackY);
   fChain->SetBranchAddress("TrackZ", &TrackZ, &b_TrackZ);
   fChain->SetBranchAddress("TrackMomentumX", &TrackMomentumX, &b_TrackMomentumX);
   fChain->SetBranchAddress("TrackMomentumY", &TrackMomentumY, &b_TrackMomentumY);
   fChain->SetBranchAddress("TrackMomentumZ", &TrackMomentumZ, &b_TrackMomentumZ);
   fChain->SetBranchAddress("TrackLocal1", &TrackLocal1, &b_TrackLocal1);
   fChain->SetBranchAddress("TrackLocal2", &TrackLocal2, &b_TrackLocal2);
   Notify();
}

Bool_t trkvalid::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void trkvalid::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t trkvalid::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef trkvalid_cxx
