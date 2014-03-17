/**
 * @file HoughLines_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>



using namespace cv;
using namespace std;

/// Global variables

/** General variables */
Mat src, edges;
Mat src_gray;
Mat standard_hough, probabilistic_hough;
int min_threshold = 2;
int max_trackbar = 35;

const char* standard_name = "Standard Hough Lines Demo";
const char* probabilistic_name = "Probabilistic Hough Lines Demo";

int s_trackbar = max_trackbar;
int p_trackbar = max_trackbar;

/// Function Headers
void help();
void Standard_Hough( int, void* );
void Probabilistic_Hough( int, void* );
void createImg_points(Mat& inputImage, string source_path);

/**
 * @function main
 */
int main( int, char** argv )
{
   /// Read the image
   /*src = imread( argv[1], 1 );

   if( src.empty() )
     { help();
       return -1;
     }*/
   
   //init images
   createImg_points(edges, "atlas_track_GPU/random1.txt");

   /// Pass the image to gray
   //cvtColor( src, src_gray, COLOR_RGB2GRAY );

   /// Apply Canny edge detector
   //Canny( src_gray, edges, 50, 200, 3 );

   /// Create Trackbars for Thresholds
   char thresh_label[50];
   sprintf( thresh_label, "Thres: %d + input", min_threshold );

   namedWindow( standard_name, WINDOW_AUTOSIZE );
   createTrackbar( thresh_label, standard_name, &s_trackbar, max_trackbar, Standard_Hough);

   namedWindow( probabilistic_name, WINDOW_AUTOSIZE );
   createTrackbar( thresh_label, probabilistic_name, &p_trackbar, max_trackbar, Probabilistic_Hough);

   /// Initialize
   Standard_Hough(0, 0);
   Probabilistic_Hough(0, 0);
   waitKey(0);
   return 0;
}

/**
 * @function help
 * @brief Indications of how to run this program and why is it for
 */
void help()
{
  printf("\t Hough Transform to detect lines \n ");
  printf("\t---------------------------------\n ");
  printf(" Usage: ./HoughLines_Demo <image_name> \n");
}

/**
 * @function Standard_Hough
 */
void Standard_Hough( int, void* )
{
  vector<Vec2f> s_lines;
  cvtColor( edges, standard_hough, COLOR_GRAY2BGR );

  /// 1. Use Standard Hough Transform
  HoughLines( edges, s_lines, 1, CV_PI/180, min_threshold + s_trackbar, 0, 0 );

  /// Show the result
  for( size_t i = 0; i < s_lines.size(); i++ )
     {
      float r = s_lines[i][0], t = s_lines[i][1];
      double cos_t = cos(t), sin_t = sin(t);
      double x0 = r*cos_t, y0 = r*sin_t;
      double alpha = 1000;

       Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
       Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
       line( standard_hough, pt1, pt2, Scalar(255,0,0), 3);
     }

   imshow( standard_name, standard_hough );
   cout << "Standard Hough CPU Found : " << s_lines.size() << endl;
}

/**
 * @function Probabilistic_Hough
 */
void Probabilistic_Hough( int, void* )
{
  vector<Vec4i> p_lines;
  cvtColor( edges, probabilistic_hough, COLOR_GRAY2BGR );
  
  //prendiamo dimenioni dell'immagine
  Size s = probabilistic_hough.size();
  //TODO: TROVARE IL CENTRO DEL CERCHIO-RILEVATORE??
  

  /// 2. Use Probabilistic Hough Transform  
  HoughLinesP(edges, p_lines, 1, CV_PI / 180, min_threshold + p_trackbar, 300 ,200);
  /// Show the result
  for( size_t i = 0; i < p_lines.size(); i++ )
     {
       Vec4i l = p_lines[i];
       line( probabilistic_hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3);
     }

   imshow( probabilistic_name, probabilistic_hough );
   cout << "Probabilistic Hough CPU Found : " << p_lines.size() << endl;
}

//funzione per la creazione dell'immagine a partire da un file di testo (lista dei punti)
void createImg_points(Mat& inputImage, string source_path){
  
  ifstream input_f;
  
  string line;
  string value;
  
  stringstream ss;
  unsigned int val_iter;
  
  vector<float> r_values;
  vector<float> phi_values;
  
  unsigned int img_dimx = 0;
  unsigned int img_dimy = 0;
  
  input_f.open(source_path);
  
  if (input_f.is_open())
  {
    while ( getline (input_f,line) )
    {
      val_iter = 0;
      ss.str(line);
      //prendiamo dati direttamente dal file ASCII in input
      while(ss >> value){
	if(val_iter > 2){
	  //i valori che ci interessano sono rho e theta (val_iter == 3 e val_iter == 4)
	  if (val_iter == 3) r_values.push_back(atof(value.c_str()));
	  else if (val_iter == 4) phi_values.push_back(atof(value.c_str()));
	}  
	val_iter++;
	
      }
      ss.clear();
    }
    input_f.close();
  }
    
  vector<Point2f> cartesianCoordinates;
  vector<float> x_values;
  vector<float> y_values;
  
  if(r_values.size() == phi_values.size()  ){
    
    for(unsigned int i = 0; i < r_values.size(); i++){     
      //convertiamo coordinate da polari a cartesiane
      float x = 0.f; float y = 0.f;
      //applichiamo la conversione standard
      //x = r * cos(theta); y = r*sin(theta)
      x = r_values.at(i) * cos(phi_values.at(i));
      y = r_values.at(i) * sin(phi_values.at(i));
      
      x_values.push_back(x);
      y_values.push_back(y);      
    }
    
    //troviam MINIMO e MASSIMO valore in x e y per stabilire la dimensione della matrice da creare per l'immagine binaria
    vector<float>::iterator result1 = min_element(begin(x_values), end(x_values));
    cout << "x_values min element at: " << distance(begin(x_values), result1) << "(" << x_values.at(distance(begin(x_values), result1)) << ")\n"; 
    vector<float>::iterator result2 = min_element(begin(y_values), end(y_values));
    cout << "y_values min element at: " << distance(begin(y_values), result2) << "(" << y_values.at(distance(begin(y_values), result2)) << ")\n";

    vector<float>::iterator result3 = max_element(begin(x_values), end(x_values));
    cout << "x_values max element at: " << distance(begin(x_values), result3) << "(" << x_values.at(distance(begin(x_values), result3)) << ")\n"; 
    vector<float>::iterator result4 = max_element(begin(y_values), end(y_values));
    cout << "y_values max element at: " << distance(begin(y_values), result4) << "(" << y_values.at(distance(begin(y_values), result4)) << ")\n";
    
    //calcoliamo la dimensione dell'immagine
    img_dimx = ceil(x_values.at(distance(begin(x_values), result3)) + abs(x_values.at(distance(begin(x_values), result1))) );
    img_dimy = ceil(y_values.at(distance(begin(y_values), result4)) + abs(y_values.at(distance(begin(y_values), result2))));
    
    for(unsigned int i = 0; i < r_values.size(); i++){
      float x = 0.f; float y = 0.f;
      //avendo coordinate negative, incrementiamo ogni punto del minimo valore rilevato nella sua dimensione
      //in modo da spostare l'origine dell'immagine finale
      x = x_values.at(i) + abs(x_values.at(distance(begin(x_values), result1)));
      y = y_values.at(i) + abs(y_values.at(distance(begin(y_values), result2)));
      
      Point2f cartesianPoint(x, y);
      cartesianCoordinates.push_back(cartesianPoint);
    }
    
  }
  
  
  //creiamo l'immagine su cui disegnare i punti (1-channel per HoughLines())
  inputImage.create(img_dimy, img_dimx, CV_8UC1);
  //e l'immagine di output (3-channel)
  //dst_cpu(img_dimy, img_dimx, CV_8UC3, CV_RGB(0,0,0));
  
  //stampiamo ogni punto
  for(unsigned int i = 0; i < cartesianCoordinates.size(); i++){
    circle(inputImage, cartesianCoordinates.at(i), 0, CV_RGB(255,255,255),3);
    //circle(dst_cpu, cartesianCoordinates.at(i),1, CV_RGB(255,255,255),3);
  }
}
