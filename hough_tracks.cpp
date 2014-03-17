#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"


int main()
{
  
  
  std::ifstream input_f;
  
  std::string line;
  std::string value;
  
  std::stringstream ss;
  unsigned int val_iter;
  
  std::vector<float> r_values;
  std::vector<float> phi_values;
  
  unsigned int img_dimx = 0;
  unsigned int img_dimy = 0;
  
  input_f.open("atlas_track_GPU/random1.txt");
  
  if (input_f.is_open())
  {
    while ( std::getline (input_f,line) )
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
    
  std::vector<cv::Point2f> cartesianCoordinates;
  std::vector<float> x_values;
  std::vector<float> y_values;
  
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
    std::vector<float>::iterator result1 = std::min_element(std::begin(x_values), std::end(x_values));
    std::cout << "x_values min element at: " << std::distance(std::begin(x_values), result1) << "(" << x_values.at(std::distance(std::begin(x_values), result1)) << ")\n"; 
    std::vector<float>::iterator result2 = std::min_element(std::begin(y_values), std::end(y_values));
    std::cout << "y_values min element at: " << std::distance(std::begin(y_values), result2) << "(" << y_values.at(std::distance(std::begin(y_values), result2)) << ")\n";

    std::vector<float>::iterator result3 = std::max_element(std::begin(x_values), std::end(x_values));
    std::cout << "x_values max element at: " << std::distance(std::begin(x_values), result3) << "(" << x_values.at(std::distance(std::begin(x_values), result3)) << ")\n"; 
    std::vector<float>::iterator result4 = std::max_element(std::begin(y_values), std::end(y_values));
    std::cout << "y_values max element at: " << std::distance(std::begin(y_values), result4) << "(" << y_values.at(std::distance(std::begin(y_values), result4)) << ")\n";
    
    //calcoliamo la dimensione dell'immagine
    img_dimx = std::ceil(x_values.at(std::distance(std::begin(x_values), result3)) + std::abs(x_values.at(std::distance(std::begin(x_values), result1))) );
    img_dimy = std::ceil(y_values.at(std::distance(std::begin(y_values), result4)) + std::abs(y_values.at(std::distance(std::begin(y_values), result2))));
    
    for(unsigned int i = 0; i < r_values.size(); i++){
      float x = 0.f; float y = 0.f;
      //avendo coordinate negative, incrementiamo ogni punto del minimo valore rilevato nella sua dimensione
      //in modo da spostare l'origine dell'immagine finale
      x = x_values.at(i) + std::abs(x_values.at(std::distance(std::begin(x_values), result1)));
      y = y_values.at(i) + std::abs(y_values.at(std::distance(std::begin(y_values), result2)));
      
      cv::Point2f cartesianPoint(x, y);
      cartesianCoordinates.push_back(cartesianPoint);
    }
    
  }
  
  
  //creiamo l'immagine su cui disegnare i punti (1-channel per HoughLines())
  cv::Mat inputImage(img_dimy, img_dimx, CV_8UC1 );
  //e l'immagine di output (3-channel)
  cv::Mat dst_cpu(img_dimy, img_dimx, CV_8UC3, CV_RGB(0,0,0));
  
  //stampiamo ogni punto
  for(unsigned int i = 0; i < cartesianCoordinates.size(); i++){
    cv::circle(inputImage, cartesianCoordinates.at(i),1, CV_RGB(255,255,255),3);
    cv::circle(dst_cpu, cartesianCoordinates.at(i),1, CV_RGB(255,255,255),3);
  }
  
  
  cv::imshow("bin input", inputImage);
    
  
  
  //TODO: trova parametri appropriati.
  // provare con "sezioni" (radianti) del plot del rilevatore?
  int maxLineGap = 200;
  int minLineLength = 60;

  std::vector<cv::Vec4i> lines_cpu;
  
  {
    const int64 start = cv::getTickCount();

    cv::HoughLinesP(inputImage, lines_cpu, 1, CV_PI / 90, 35, minLineLength, maxLineGap);	
    //cv::HoughLines(inputImage, lines_cpu, 1, CV_PI / 90, 20, 0, 0);

    const double timeSec = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "CPU Time : " << timeSec * 1000 << " ms" << std::endl;
    std::cout << "CPU Found : " << lines_cpu.size() << std::endl;
  }

  for (size_t i = 0; i < lines_cpu.size(); ++i)
  {
      cv::Vec4i l = lines_cpu[i];
      cv::line(dst_cpu, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 3, CV_AA);
  }
  
  cv::imshow("hough output", dst_cpu);
  
  cv::waitKey(-1);
  
  return 0;
}