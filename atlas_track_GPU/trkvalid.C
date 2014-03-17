#define trkvalid_cxx
#include "trkvalid.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void trkvalid::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L trkvalid.C
//      Root > trkvalid t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
  
  ofstream outfile(m_output_file.c_str(), std::ofstream::out);

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      // 
      outfile << RecQoverP << "  " << RecPhi0 << "  " << RecD0 << "  " << RecZ0<< "  " << RecTheta;
      int id_max = 0;
      int idh = 0;
      for(vector<float>::iterator id=HitX->begin();id!=HitX->end();++id,++idh){
	if((HitRadius)->at(idh)>550) continue;
	++id_max;
      }
      outfile << "  " << id_max << endl;
      idh = 0;
      for(vector<float>::iterator id=HitX->begin();id!=HitX->end();++id,++idh){
	if((HitRadius)->at(idh)>550) continue;
	outfile << "  " << (HitX)->at(idh) << "  " <<  (HitY)->at(idh) << "  " << (HitZ)->at(idh) << "  " << (HitRadius)->at(idh) << "  " <<  (HitPhi)->at(idh) << endl; 
      }
   }
   
   outfile.close();

}
