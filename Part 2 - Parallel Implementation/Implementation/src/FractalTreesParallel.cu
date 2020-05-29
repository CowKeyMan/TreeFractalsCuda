#include "shared/jbutil.h"
#include <assert.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <map>
#include <climits>

using std::cout;
using std::endl;
using std::cerr;
using std::endl;
using std::string;

using jbutil::matrix;
using jbutil::image;

__global__ void populate_sin_cos_maps(float *sin_map, float *cos_map){
  float rad = threadIdx.x * 0.01745329251f; // small number is pi/180
  sincosf(rad, &sin_map[threadIdx.x], &cos_map[threadIdx.x]);
}

__global__ void calculate_points(
  short *angles,
  float *pointsX,
  float *pointsY,
  const int iterations,
  const float length_multiplier,
  const int rotation_angle_degrees,
  float *_sin_map,
  float *_cos_map
)
{
  __shared__ float sin_map[512];
  __shared__ float cos_map[512];
  sin_map[threadIdx.x] = _sin_map[threadIdx.x];
  cos_map[threadIdx.x] = _cos_map[threadIdx.x];
  __syncthreads();

  float line_length = 1*length_multiplier;

  const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = 0; i < iterations; ++i){
    if( index < (1 << i) ){ // first we compute first 2, then next 4, then next 8, etc
      unsigned int array_index = index*2 + (2<<i); // first 2 already initialized
      unsigned int array_index_plus_1 = array_index + 1; // first 2 already initialized
      unsigned int array_index_div_2 = array_index >> 1;

      angles[array_index] = ( (angles[array_index_div_2] + rotation_angle_degrees) ) % 360;
      angles[array_index_plus_1] = ( (angles[array_index_div_2] - rotation_angle_degrees) + 360 ) % 360;

      pointsX[array_index] = pointsX[array_index_div_2] + line_length * sin_map[ (angles[array_index]) ];
      pointsY[array_index] = pointsY[array_index_div_2] + line_length * cos_map[ (angles[array_index]) ];
      pointsX[array_index_plus_1] = pointsX[array_index_div_2] + line_length * sin_map[ (angles[array_index_plus_1]) ];
      pointsY[array_index_plus_1] = pointsY[array_index_div_2] + line_length * cos_map[ (angles[array_index_plus_1]) ];
    }
    line_length *= length_multiplier;
    __syncthreads();
  }
}

__global__ void calculate_points_single_iteration(
  short *angles,
  float *pointsX,
  float *pointsY,
  float line_length,
  const float length_multiplier,
  const int rotation_angle_degrees,
  float *sin_map,
  float *cos_map,
  unsigned long start_index
)
{
  const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned long array_index = index*2 + start_index; // first 2 already initialized
  unsigned long array_index_plus_1 = array_index + 1; // first 2 already initialized
  unsigned long array_index_div_2 = array_index >> 1;

  angles[array_index] = ( (angles[array_index_div_2] + rotation_angle_degrees) ) % 360;
  angles[array_index_plus_1] = ( (angles[array_index_div_2] - rotation_angle_degrees) + 360 ) % 360;

  pointsX[array_index] = pointsX[array_index_div_2] + line_length * sin_map[ (angles[array_index]) ];
  pointsY[array_index] = pointsY[array_index_div_2] + line_length * cos_map[ (angles[array_index]) ];
  pointsX[array_index_plus_1] = pointsX[array_index_div_2] + line_length * sin_map[ (angles[array_index_plus_1]) ];
  pointsY[array_index_plus_1] = pointsY[array_index_div_2] + line_length * cos_map[ (angles[array_index_plus_1]) ];
}

__global__ void calculateMin(float *points, float *storeList, float *retValue, const int iterations, unsigned long no_of_threads){
  const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < no_of_threads){
    storeList[index] = points[ index*2 + (points[index*2] > points[index*2 + 1]) ];
  }

  no_of_threads /= 2;
  __syncthreads();

  for(int i = 1; i < iterations; ++i){ // start i from 1 as the first iteration has already been done
    if(index < no_of_threads){
      storeList[index] = storeList[ index*2 + (storeList[index*2] > storeList[index*2 + 1]) ];
    }
    no_of_threads/=2;
    __syncthreads();
  }

  *retValue = storeList[0];
}

__global__ void calculateMin_single_iteration(float *points, float *storeList){
  const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
  storeList[index] = points[ index*2 + (points[index*2] > points[index*2 + 1]) ];
}

__global__ void calculateMax(float *points, float *storeList, float *retValue, const int iterations, unsigned long no_of_threads){
  const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < no_of_threads){
    storeList[index] = points[ index*2 + (points[index*2] < points[index*2 + 1]) ];
    //storeList[index] = fmaxf(points[index*2], points[index*2 + 1]);
  }

  no_of_threads /= 2;
  __syncthreads();

  for(int i = 1; i < iterations; ++i){ // start i from 1 as the first iteration has already been done
    if(index < no_of_threads){
      storeList[index] = storeList[ index*2 + (storeList[index*2] < storeList[index*2 + 1]) ];
      //storeList[index] = fmaxf(points[index*2], points[index*2 + 1]);
    }
    no_of_threads/=2;
    __syncthreads();
  }

  *retValue = storeList[0];
}

__global__ void calculateMax_single_iteration(float *points, float *storeList){
  const unsigned long index = blockIdx.x * blockDim.x + threadIdx.x;
  storeList[index] = points[ index*2 + (points[index*2] < points[index*2 + 1]) ];
  //storeList[index] = fmaxf(points[index*2], points[index*2 + 1]);
}

__global__ void map_points_to_pixels(float *pointsX, float *pointsY, const float x_mul, const float x_add, const float y_mul, const float y_add){
  const unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  pointsX[i] = pointsX[i] * x_mul + x_add;
  pointsY[i] = pointsY[i] * y_mul + y_add;
}

__global__ void initialize_all_to_zero(short *m_image, int max_size){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < max_size){
    m_image[index] = 255;
  }
}

__global__ void draw_points(const float *pointsX, const float *pointsY, short *m_image, const int image_height){
  const unsigned long index = blockIdx.y * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long index_div_2 = index/2;

  float px1 = pointsX[index];
  float px2 = pointsX[index_div_2];
  float py1 = pointsY[index];
  float py2 = pointsY[index_div_2];

  // absolute difference without elif
  float diffX = (px1-px2)*(px1>=px2) + (px2-px1)*(px2>px1);
  float diffY = (py1-py2)*(py1>=py2) + (py2-py1)*(py2>py1);

		// last multiplication essentially gets -1 or 1 (or 0 if difference is 0) to see if we need to increase or decrease
  float x_increase = ( 1*(diffX>=diffY) + (diffX/(diffY+(diffY==0)))*(diffX<diffY) ) * ( (px2-px1)/ (diffX+(diffX==0)) );
  float y_increase = ( (diffY/(diffX+(diffX==0)))*(diffX>=diffY) + 1*(diffX<diffY) ) * ( (py2-py1)/(diffY+(diffY==0)) );

  float startx = round(px1) * (diffX>=diffY) + px1 * (diffX<diffY);
  float starty = py1 * (diffX>=diffY) + round(py1) * (diffX<diffY);

  float endx = round(px2);
  float endy = round(py2);

  float x = startx, y = starty;
  for(; ((diffX>=diffY) & (round(x) != endx)) | ((diffY>diffX) & (round(y) != endy)); x+=x_increase, y+=y_increase){
    m_image[lround(x) * image_height + lround(y)] = 0;
  }
  m_image[lround(x) * image_height + lround(y)] = 255 - 255*(index != 0);
}

int main(int argc, char *argv[]){
  if (argc != 6)
  {
    cout << "Error: wrong number of parameters\n"
         << "Usage:\n\t<output image width>\n\t<output image height>\n\t"
         << "<length multiplier per iteration>\n\t<rotation per iteration (in degrees) - between 0 and 180>\n\t"
         << "<number of iterations - between 1 and 26>" << endl;
    exit(1);
  }

  int image_width = atoi(argv[1]), image_height = atoi(argv[2]);
  float length_multiplier = atof(argv[3]);
  int rotation_angle_degrees = atoi(argv[4]);
  int iterations = atoi(argv[5]);

  if (0 > rotation_angle_degrees || rotation_angle_degrees > 180){
    failwith("Please enter a number of angle degrees between 0 and 180 (both included)");
  }

  if(0 > iterations || iterations > 26){
    failwith("Please enter a number of iterations between 1 and 26(both included)"); // otherwise a memory overflow occurs
  }

  /* for testing
  int image_width=1024, image_height=512;
  float length_multiplier = 1.1; // (multiply the current line's length by this number
  int rotation_angle_degrees = 30; // (The amount to rotate per iteration) must be between 0 and 180
  int iterations = 15; // (number of iterations) Must be between 1 and 26 (both included) otherwise it uses too much memory, as memory usage is 10x2^iterations bytes
  */

  const unsigned long no_of_points = (1 << (iterations)); // 2^iterations lines + the original line

  // Declare the line length (after the first 11 iterations, since we use an internal line length in the gpu functions)
  float line_length = 1;
  for(int i = 0; i < 11; ++i){
    line_length*=length_multiplier;
  }

  // Declare sin and cosine maps
  float *sin_map, *cos_map;
  const int map_physical_size = 512 * sizeof(float);

  // Declare angles and pointsX and pointsY lists (stored within the gpu)
  short *angles;
  float *pointsX, *pointsY;
  const unsigned long short_list_size = no_of_points * sizeof(short);
  const unsigned long float_list_size = no_of_points * sizeof(float);

  // initilize the first 2 of angles, pointsX and pointsY
  short angles_host[] = {0, 0};
  float pointsX_host[] = {0, 0};
  float pointsY_host[] = {0, 1};

  float *minMax_X_Y; //Order: maxX, minY, maxY (we do not need minX as this is just maxX*-1)
  float *minMaxWorkingList;

  // --------- START TIMING PART 1 ----------
  double t = jbutil::gettime();

  cudaMalloc((void**) &sin_map, map_physical_size);
  cudaMalloc((void**) &cos_map, map_physical_size);

  cudaMalloc((void**) &angles, short_list_size);
  cudaMalloc((void**) &pointsX, float_list_size);
  cudaMalloc((void**) &pointsY, float_list_size);

  cudaMemcpy(angles, angles_host, 2*sizeof(short), cudaMemcpyHostToDevice);
  cudaMemcpy(pointsX, pointsX_host, 2*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(pointsY, pointsY_host, 2*sizeof(float), cudaMemcpyHostToDevice);

  populate_sin_cos_maps<<<1, 512>>>(sin_map, cos_map);

  // Calculate the new points which the lines will connect to
  // Calculate the first 2048
  calculate_points<<<1, 512>>>(
    angles,
    pointsX,
    pointsY,
    ((iterations-1 < 10)? iterations-1 : 10),
    length_multiplier,
    rotation_angle_degrees,
    sin_map,
    cos_map
  );
  // calculate the rest, iteration by iteration
  for(int i = 2048; i < no_of_points; i *= 2){
    calculate_points_single_iteration<<<i/1024, 512>>>(
      angles,
      pointsX,
      pointsY,
      line_length,
      length_multiplier,
      rotation_angle_degrees,
      sin_map,
      cos_map,
      i
    );
  }

  cudaFree(sin_map);
  cudaFree(cos_map);
  cudaFree(angles);

  cudaMalloc((void**) &minMax_X_Y, sizeof(float) * 3);
  cudaMalloc((void**) &minMaxWorkingList, float_list_size/2);
  
		// Find Max X
  bool firstTime = true;
  for(int i = no_of_points/2; i > 512; i /= 2){
    if(firstTime){
      calculateMax_single_iteration<<<i/512, 512>>>(pointsX, minMaxWorkingList);
      firstTime = false;
    }else{
      calculateMax_single_iteration<<<i/512, 512>>>(minMaxWorkingList, minMaxWorkingList);
    }
  }
  if(firstTime){
    calculateMax<<<1, 512>>>(pointsX, minMaxWorkingList, &minMax_X_Y[0], iterations, no_of_points/2);
  }else{
    calculateMax<<<1, 512>>>(minMaxWorkingList, minMaxWorkingList, &minMax_X_Y[0], iterations, no_of_points/2);
  }
  // Find Min Y
  firstTime = true;
  for(int i = no_of_points/2; i > 512; i /= 2){
    if(firstTime){
      calculateMin_single_iteration<<<i/512, 512>>>(pointsY, minMaxWorkingList);
      firstTime = false;
    }else{
      calculateMin_single_iteration<<<i/512, 512>>>(minMaxWorkingList, minMaxWorkingList);
    }
  }
  if(firstTime){
    calculateMin<<<1, 512>>>(pointsY, minMaxWorkingList, &minMax_X_Y[1], iterations, no_of_points/2);
  }else{
    calculateMin<<<1, 512>>>(minMaxWorkingList, minMaxWorkingList, &minMax_X_Y[1], iterations, no_of_points/2);
  }

  // Find Max Y
  firstTime = true;
  for(int i = no_of_points/2; i > 512; i /= 2){
    if(firstTime){
      calculateMax_single_iteration<<<i/512, 512>>>(pointsY, minMaxWorkingList);
      firstTime = false;
    }else{
      calculateMax_single_iteration<<<i/512, 512>>>(minMaxWorkingList, minMaxWorkingList);
    }
  }
  if(firstTime){
    calculateMax<<<1, 512>>>(pointsY, minMaxWorkingList, &minMax_X_Y[2], iterations, no_of_points/2);
  }else{
    calculateMax<<<1, 512>>>(minMaxWorkingList, minMaxWorkingList, &minMax_X_Y[2], iterations, no_of_points/2);
  }
  
		float maxX_minY_maxY[3];
  cudaMemcpy(maxX_minY_maxY, minMax_X_Y, sizeof(float) * 3, cudaMemcpyDeviceToHost);

  cudaFree(minMax_X_Y);
  cudaFree(minMaxWorkingList);
  
		float x_mul = (maxX_minY_maxY[0] == 0)? 1: (image_width-1)/(maxX_minY_maxY[0]*2);
  float x_add = (image_width-1)/2.0f;
  float y_mul = (maxX_minY_maxY[2]==maxX_minY_maxY[1])? (image_height-1) : -(image_height-1)/(maxX_minY_maxY[2] - maxX_minY_maxY[1]);
  float y_add = (maxX_minY_maxY[2]==maxX_minY_maxY[1])? 0 : (image_height-1) + (image_height-1)/(maxX_minY_maxY[2]-maxX_minY_maxY[1]) * maxX_minY_maxY[1];

  unsigned int blocks = no_of_points/512 + (no_of_points % 512 != 0);
  map_points_to_pixels<<<blocks, ((no_of_points<512)? no_of_points : 512)>>>(pointsX, pointsY, x_mul, x_add, y_mul, y_add);

  // --------- STOP TIMING PART 1 ----------
  t = jbutil::gettime() - t;
  std::cerr << "Time taken to generate the points: " << t << "s" << endl;


  short *m_image;

  // --------- START TIMING PART 2 ----------
  t = jbutil::gettime();

  cudaMalloc((void**) &m_image, sizeof(short) * image_height * image_width);

  initialize_all_to_zero<<<(image_width*image_height)/512 + ((image_width*image_height)%512 != 0), 512>>>(m_image, image_width*image_height);
  draw_points<<<(no_of_points<512)? 1 : no_of_points/512, ((no_of_points<512)? no_of_points : 512)>>>(pointsX, pointsY, m_image, image_height);

  // copy from device to host
  short *m_image_host = (short*)malloc(sizeof(short)*image_width*image_height);
  cudaMemcpy(m_image_host, m_image, sizeof(short)*image_width*image_height, cudaMemcpyDeviceToHost);
  matrix<int> m_image_2;

  m_image_2.resize(image_height, image_width);

  // copy from flattened array to image matrix
  for(int x = 0; x < image_width*image_height; ++x){
      m_image_2(x%image_height, x/image_height) = m_image_host[x];
  }

  //Clean up
  cudaFree(pointsX);
  cudaFree(pointsY);
  cudaFree(m_image);
  free(m_image_host);

  image<int> image_out = image<int>(image_height, image_width, 1, 255);
  image_out.set_channel(0, m_image_2);

  // --------- STOP TIMING PART 2 ----------
  t = jbutil::gettime() - t;
  std::cerr << "Time taken to create the image: " << t << "s" << endl;

  //SAVE
  char outfile[150];
  sprintf(outfile, "output_images/%d x%d _m=%.2f _theta=%d _n= %.0d.pgm",
    image_width, image_height,
    length_multiplier,
    rotation_angle_degrees,
    iterations
  );
  std::ofstream file_out(outfile);
  image_out.save(file_out);
}
//cudaDeviceSynchronize(); size_t fr, tot; cudaMemGetInfo(&fr, &tot); cout << "mem " << fr << " " << tot << endl;
//cudaError_t error = cudaGetLastError();if(error != cudaSuccess){printf("CUDA error1: %s\n", cudaGetErrorString(error));exit(-1);}
