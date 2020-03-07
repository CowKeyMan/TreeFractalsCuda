#ifndef __workingWithDegrees_h
#define __workingWithDegrees_h

#include "jbutil.h"
#include <cmath>

float sin_map[360]; // maps angle i to its sin at index i+180 (-180 <= i <= 180)
float cos_map[360]; // maps angle i to its cos at index i+180 (-180 <= i <= 180)



float degrees_to_radians(int degrees){
		return degrees * pi / 180;
}

void populate_sin_map(float* sin_map){
		for(int i = -180; i < 179; i++){
				sin_map[i+180] = sin(degrees_to_radians(i));
		}
}

void populate_cos_map(float* cos_map){
		for(int i = -180; i < 179; i++){
				cos_map[i+180] = cos(degrees_to_radians(i));
		}
}

#endif
