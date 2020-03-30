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
		float rad = (threadIdx.x-180.0f) * 0.01745329251f; // small number is pi/180
	 sincosf(rad, &sin_map[threadIdx.x], &cos_map[threadIdx.x]);
}

int main(int argc, char *argv[]){
		float *sin_map, *cos_map;
		float map_size = (360 + 360%32);
		float map_physical_size = map_size * sizeof(float);
		cudaMalloc((void**) &sin_map, map_physical_size);
		cudaMalloc((void**) &cos_map, map_physical_size);
		
		populate_sin_cos_maps<<<1, map_size>>>(sin_map, cos_map);

		float s[362], c[362];
		cudaMemcpy(s, sin_map, map_physical_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(c, cos_map, map_physical_size, cudaMemcpyDeviceToHost);

		for(int i = 0; i < 362; ++i){
				cout << i-180 << " " << s[i] << " " << c[i] << endl;
		}

		cudaFree(sin_map);
  cudaFree(cos_map);
}
