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

//__global__ void populate_sin_cos_maps(*sin_map, *cos_map);
__global__ void populate_sin_cos_maps(float *sin_map, float *cos_map){
		float rad = threadIdx.x * 0.01745329251f; // small number is pi/180
	 sincosf(rad, &sin_map[threadIdx.x], &cos_map[threadIdx.x]);
}

__global__ void calculate_points(
		short *angles,
		float *pointsX,
		float *pointsY,
		const int iterations,
		const float initial_line_length,
		const float length_multiplier_,
		const int rotation_angle_degrees
)
{
		// initialize line_length
		__shared__ float line_length;
		__shared__ float length_multiplier;
		line_length = initial_line_length;
		length_multiplier = length_multiplier_;

		// initialize the cos and sin maps (Note, blockdim must be greater than 360) 
		__shared__ float sin_map[512];
		__shared__ float cos_map[512];
		float rad = threadIdx.x * 0.01745329251f; // small number is pi/180
		sincosf(rad, &sin_map[threadIdx.x], &cos_map[threadIdx.x]);
		__syncthreads();

		// initialize the angles and points list
		const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		for(int i = 0; i < 4; ++i){
				if( index < (1 << i) ){ // first we compute first 2, then next 4, then next 8, etc
						unsigned int array_index = index*2 + (2<<i); // first 2 already initialized
						unsigned int array_index_plus_1 = array_index + 1; // first 2 already initialized
						unsigned int array_index_div_2 = array_index >> 1;

						angles[array_index] = ( (angles[array_index_div_2] - rotation_angle_degrees) + 360 ) % 360;
						angles[array_index_plus_1] = ( (angles[array_index_div_2] + rotation_angle_degrees) + 360 ) % 360;

						pointsX[array_index] = pointsX[array_index_div_2] + line_length * sin_map[ angles[array_index]+180 ];
						pointsY[array_index] = pointsY[array_index_div_2] + line_length * cos_map[ angles[array_index]+180 ];
						pointsX[array_index_plus_1] = pointsX[array_index_div_2] + line_length * sin_map[ angles[array_index_plus_1]+180 ];
						pointsY[array_index_plus_1] = pointsY[array_index_div_2] + line_length * cos_map[ angles[array_index_plus_1]+180 ];
				}
				__syncthreads();
				line_length *= length_multiplier;
				__syncthreads();
		}
}

int main(int argc, char *argv[]){
		float length_multiplier = 0.5; // (multiply the current line's length by this number
		int rotation_angle_degrees = 90; // (The amount to rotate per iteration) must be between 0 and 180
		int iterations = 5; // (number of iterations) Must be between 1 and 26 (both included) otherwise it uses too much memory, as memory usage is 10x2^iterations bytes

		const float line_length = 1;

		const unsigned long no_of_points = (2 << (iterations - 1));

		// Declare sin and cosine maps
		/*float *sin_map, *cos_map;
		const int map_size = (360 + 360%32);
		const int map_physical_size = map_size * sizeof(float);
		cudaMalloc((void**) &sin_map, map_physical_size);
		cudaMalloc((void**) &cos_map, map_physical_size);
		*/
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
		float pointsY_host[] = {0, line_length};
		cudaMemcpy(angles, angles_host, 2*sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(pointsX, pointsX_host, 2*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(pointsY, pointsY_host, 2*sizeof(float), cudaMemcpyHostToDevice);

		//populate_sin_cos_maps<<<1, map_size>>>(sin_map, cos_map);

		calculate_points<<<1, 512>>>(angles, pointsX, pointsY, iterations-1, line_length/2, length_multiplier, rotation_angle_degrees);

		short *a = (short*)malloc(short_list_size);
		float *px = (float*)malloc(float_list_size);
		float *py = (float*)malloc(float_list_size);
		cudaMemcpy(a, angles, short_list_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(px, pointsX, float_list_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(py, pointsY, float_list_size, cudaMemcpyDeviceToHost);



		

		for(int i = 0; i < no_of_points; ++i){
				cout << i << " " << a[i] << " " << px[i] << " " << py[i] << endl;
		}
}
