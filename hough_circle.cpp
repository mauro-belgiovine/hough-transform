#include <string.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream> 


using namespace std;

#define NHMAX 100
#define nbin 500

#define Amax 10000.f // mm
#define Bmax 10000.f // mm
#define Rmax (sqrt(pow(Amax+500.f, 2) + pow(Bmax+500.f, 2))) //mm
#define Rmin 0.f// mm

#define Amin ((-1)*Amax)
#define Bmin ((-1)*Bmax)

int ***acc_Mat;

float dA = (Amax - Amin)/nbin;
float dB = (Bmax - Bmin)/nbin;
float dR = (Rmax-Rmin)/nbin;

vector<float> x_values;
vector<float> y_values;

void hough_circle();
void read_inputFile(string file_path);

int main(int argc, char* argv[]){
  
  float R = 0.f;
  
  acc_Mat = (int ***) malloc(sizeof(int **) * nbin);
  assert(acc_Mat != NULL);
  for(unsigned int i = 0; i < nbin; i++){
    acc_Mat[i] = (int **) malloc(sizeof(int *) * nbin);
    assert(acc_Mat[i] != NULL);
    for(unsigned int y = 0; y < nbin; y++){
      acc_Mat[i][y] = (int *) malloc(sizeof(int) * nbin);
      assert(acc_Mat[i][y] != NULL);
      memset((void *) acc_Mat[i][y], 0, sizeof(int)* nbin );
    }
  }
    
  cout << "dA " << dA << " dB " << dB << " dR " << dR << endl;
  
  //memset(&acc_Mat, 0, (sizeof(int)*(nbin*nbin*nbin)) );
  //riempi i valori dentro x_values e y_values
  read_inputFile("atlas_track_GPU/hits2.txt");
  
  for(unsigned int i = 0; i < x_values.size(); i++){
    
    //cout << x_values.at(i) << " " << y_values.at(i) << endl;
    
    for(int ia = 0; ia < nbin; ia++){
      
      float a = Amin + (ia + 0.5f)*dA;
      
      for(int ib = 0; ib < nbin; ib++){
	
	  float b = Bmin + (ib + 0.5f)*dB;
	  
	  R = sqrt(pow(a-x_values.at(i), 2) + pow(b-y_values.at(i),2));
	  
	  int iR = (nbin*(R-Rmin)/(Rmax - Rmin));
	  
	  if (R < Rmax) acc_Mat[ia][ib][iR] += 1;
	  
	  //cout << ia << " " << ib << " " << iR << " = " << acc_Mat[ia][ib][iR] << endl;
      }
      
    }
    
  }
  
  //trova il massimo
  
  int accumax = -1;
  int iaMax = 0;
  int ibMax = 0;
  int iRMax = 0;
  
  for(unsigned int ia = 0; ia < nbin; ia++){
    
    for(unsigned int ib = 0; ib < nbin; ib++){
      
      for(unsigned int iR = 0; iR < nbin; iR++){
	
	if(acc_Mat[ia][ib][iR] > accumax){
	  accumax = acc_Mat[ia][ib][iR];
	  iaMax = ia;
	  ibMax = ib;
	  iRMax = iR;
	}
	
      }
      
    }
    
  }
  
  float a = Amin + (iaMax + 1) * dA; //0
  float b = Bmin + (ibMax + 1) * dB; //0
  float r = Rmin + (iRMax) * dR; //18.4
  
  cout << "a = " << a << " b = " << b << " r = " << r << endl;
  
  return 0;
}

/*****************************
 * Hough Circle detection 
 *****************************/

void hough_circle(){
  
}

void read_inputFile(string file_path)
{
  
  ifstream input_f;
  
  string line;
  string value;
  
  stringstream ss;
  unsigned int val_iter;
  
  input_f.open(file_path);
  
  if (input_f.is_open())
  {
    while ( getline (input_f,line) )
    {
      val_iter = 0;
      ss.str(line);
      //prendiamo dati direttamente dal file ASCII in input
      while(ss >> value){
	  //i valori che ci interessano sono X e Y
	  if (val_iter == 0) x_values.push_back(atof(value.c_str()));
	  else if (val_iter == 1) y_values.push_back(atof(value.c_str()));
	val_iter++;
	
      }
      ss.clear();
    }
    input_f.close();
  }else{
    cout << "file not found!" << endl;
    exit(1);
  }
  
  
  
  
}


