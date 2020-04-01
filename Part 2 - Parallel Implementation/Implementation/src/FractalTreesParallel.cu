//#include "shared/jbutil.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <map>
#include <climits>
#include <assert.h>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

//using jbutil::matrix;
//using jbutil::image;

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
		float *sin_map,
		float *cos_map
)
{
		float line_length = 1*length_multiplier;

		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

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
				__syncthreads();
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
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

		unsigned int array_index = index*2 + start_index; // first 2 already initialized
		unsigned int array_index_plus_1 = array_index + 1; // first 2 already initialized
		unsigned int array_index_div_2 = array_index >> 1;

		angles[array_index] = ( (angles[array_index_div_2] + rotation_angle_degrees) ) % 360;
		angles[array_index_plus_1] = ( (angles[array_index_div_2] - rotation_angle_degrees) + 360 ) % 360;

		pointsX[array_index] = pointsX[array_index_div_2] + line_length * sin_map[ (angles[array_index]) ];
		pointsY[array_index] = pointsY[array_index_div_2] + line_length * cos_map[ (angles[array_index]) ];
		pointsX[array_index_plus_1] = pointsX[array_index_div_2] + line_length * sin_map[ (angles[array_index_plus_1]) ];
		pointsY[array_index_plus_1] = pointsY[array_index_div_2] + line_length * cos_map[ (angles[array_index_plus_1]) ];
}

__global__ void calculateMin(float *points, float *storeList, float *retValue, const int iterations, unsigned long no_of_threads){
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
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
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		storeList[index] = points[ index*2 + (points[index*2] > points[index*2 + 1]) ];
}

__global__ void calculateMax_single_iteration(float *points, float *storeList){
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		storeList[index] = points[ index*2 + (points[index*2] < points[index*2 + 1]) ];
}

__global__ void calculateMax(float *points, float *storeList, float *retValue, const int iterations, unsigned long no_of_threads){
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		if(index < no_of_threads){
			 storeList[index] = points[ index*2 + (points[index*2] < points[index*2 + 1]) ];
		}
		
		no_of_threads /= 2;
		__syncthreads();

		for(int i = 1; i < iterations; ++i){ // start i from 1 as the first iteration has already been done
				if(index < no_of_threads){
						storeList[index] = storeList[ index*2 + (storeList[index*2] < storeList[index*2 + 1]) ];
				}
				no_of_threads/=2;
				__syncthreads();
		}
		
		*retValue = storeList[0];
}

__global__ void map_points_to_pixels(float *pointsX, float *pointsY, const float x_mul, const float x_add, const float y_mul, const float y_add){
		const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		pointsX[i] = pointsX[i] * x_mul + x_add;
		pointsY[i] = pointsY[i] * y_mul + y_add;
}


int main(int argc, char *argv[]){
		int image_width=1024, image_height=1024;
		float length_multiplier = 1; // (multiply the current line's length by this number
		int rotation_angle_degrees = 90; // (The amount to rotate per iteration) must be between 0 and 180
		int iterations = 15; // (number of iterations) Must be between 1 and 26 (both included) otherwise it uses too much memory, as memory usage is 10x2^iterations bytes

		const unsigned long no_of_points = (1 << (iterations));

		// Declare the line length
		float line_length = 1;
		for(int i = 0; i < 11; ++i){
				line_length*=length_multiplier;
		}

		// Declare sin and cosine maps
		float *sin_map, *cos_map;
		const int map_size = (360 + 360%32);
		const int map_physical_size = map_size * sizeof(float);
		cudaMalloc((void**) &sin_map, map_physical_size);
		cudaMalloc((void**) &cos_map, map_physical_size);

		// Declare angles and pointsX and pointsY lists
		short *angles;
		float *pointsX, *pointsY;
		const unsigned long short_list_size = no_of_points * sizeof(short);
		const unsigned long float_list_size = no_of_points * sizeof(float);
		cudaMalloc((void**) &angles, short_list_size);
		cudaMalloc((void**) &pointsX, float_list_size);
		cudaMalloc((void**) &pointsY, float_list_size);

		// initilize the first 2 of angles, pointsX and pointsY
		short angles_host[] = {0, 0};
		float pointsX_host[] = {0, 0};
		float pointsY_host[] = {0, 1};
		cudaMemcpy(angles, angles_host, 2*sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(pointsX, pointsX_host, 2*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(pointsY, pointsY_host, 2*sizeof(float), cudaMemcpyHostToDevice);

		float *minMax_X_Y; //Order: minX, minY, maxY (we do not need maxX as this is just minX*-1)
		float *minMaxWorkingList;
		cudaMalloc((void**) &minMax_X_Y, sizeof(float) * 3);
		cudaMalloc((void**) &minMaxWorkingList, float_list_size/2);

		// --------- START TIMING PART 1 ----------

		populate_sin_cos_maps<<<1, map_size>>>(sin_map, cos_map);

		unsigned int blocks = no_of_points/512 + (no_of_points % 512 != 0);

		calculate_points<<<1, 512>>>(angles, pointsX, pointsY, ((iterations-1 < 10)? iterations-1 : 10), length_multiplier, rotation_angle_degrees, sin_map, cos_map);
		for(int i = 2048; i < no_of_points; i *= 2){
				calculate_points_single_iteration<<<i/1024, 512>>>(angles, pointsX, pointsY, line_length, length_multiplier, rotation_angle_degrees, sin_map, cos_map, i);
		}

		
		//cudaFree(angles);

		bool firstTime = true;
		for(int i = no_of_points; i > 1024; i /= 2){
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

	 firstTime = true;
		for(int i = no_of_points; i > 1024; i /= 2){
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

		firstTime = true;
		for(int i = no_of_points; i > 1024; i /= 2){
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


		// make the host block until the device is finished with foo
  cudaDeviceSynchronize();
		 // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }else{
				cout << "ok for now" << endl;
		}




		/*float maxX_minY_maxY[3];
		cudaMemcpy(maxX_minY_maxY, minMax_X_Y, sizeof(float) * 3, cudaMemcpyDeviceToHost);
		float x_mul = (maxX_minY_maxY[0] == 0)? 1: image_width/(maxX_minY_maxY[0]*2);
		float x_add = image_width/2.0f;
		float y_mul = (maxX_minY_maxY[2]==maxX_minY_maxY[1])? image_height : -image_height/(maxX_minY_maxY[2] - maxX_minY_maxY[1]);
		float y_add = (maxX_minY_maxY[2]==maxX_minY_maxY[1])? 0 : image_height + image_height/(maxX_minY_maxY[2]-maxX_minY_maxY[1]) * maxX_minY_maxY[1];

		map_points_to_pixels<<<blocks, ((no_of_points<512)? no_of_points : 512)>>>(pointsX, pointsY, x_mul, x_add, y_mul, y_add);*/

		// --------- STOP TIMING PART 1 ----------






		short *a = (short*)malloc(short_list_size);
		float *px = (float*)malloc(float_list_size);
		float *py = (float*)malloc(float_list_size);
		cudaMemcpy(a, angles, short_list_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(px, pointsX, float_list_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(py, pointsY, float_list_size, cudaMemcpyDeviceToHost);

		for(unsigned long i = no_of_points-100; i < no_of_points; ++i){
				cout << i << " " << a[i] << " " << px[i] << " " << py[i] << endl;
		}


		cout << endl << endl;

		float *t = (float*)malloc(float_list_size/2);
		cudaMemcpy(t, minMaxWorkingList, float_list_size/2, cudaMemcpyDeviceToHost);

		for(int i = 0; i < no_of_points/2; ++i){
				//cout << t[i] << endl;
		}

		float *minMax_X_Y_local = (float*)malloc(sizeof(float)*3);
		cudaMemcpy(minMax_X_Y_local, minMax_X_Y, sizeof(float) * 3, cudaMemcpyDeviceToHost);

		cout << endl << endl;
		for(int i = 0; i < 3; ++i){
				cout << minMax_X_Y_local[i] << endl;
		}
}
