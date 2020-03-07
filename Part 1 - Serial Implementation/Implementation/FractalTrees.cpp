#include "shared/workingWithDegrees.h"
#include "shared/jbutil.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <map>
#include <climits>

using std::cout;
using std::endl;

using jbutil::matrix;
using jbutil::image;

void calculcate_points(
		float* pointsX, float* pointsY,
		short* angles,
		const float* sin_map, const float* cos_map,
		const float line_length, const float length_decrease,
		const int rotation_angle_degrees,
		const unsigned int no_of_points
);
// Inputs:
//				int rotation_angle_degrees (The amount to rotate per iteration [must be prefectly divisible by 8, 0 < R < 180])
//				float initial_length (length of first line)
//    float length_decrease (percentge of next line in iteration)
//    iterations (number of iterations)
int main(){
		float initial_length = 50; // Initial length of first line
		float length_decrease = 0.9f; // How much line gets smaller per iteration
		int rotation_angle_degrees = 5; // How much to turn  per iteration (in degrees)
		int iterations = 19; 

		// get the number of points we will have by the end
		unsigned int no_of_points = (1 << iterations) - 1; // 2^iterations - 1

		// lists of the properties belonging to each point
		short angles[no_of_points];
		angles[0] = 0;
		float pointsX[no_of_points];
		pointsX[0] = 0;
		float pointsY[no_of_points];
		pointsY[0] = initial_length;

		cout << sizeof(angles[0]) << " " << sizeof(pointsX[0]) << endl;

		// call to use the sin and cosine maps defined in workingWithDegrees.h
		populate_sin_map(sin_map);
		populate_cos_map(cos_map);
		
		double t = jbutil::gettime();

		calculcate_points(
				pointsX, pointsY,
				angles,
				sin_map, cos_map,
				initial_length,
				length_decrease,
				rotation_angle_degrees,
				no_of_points
		);

		t = jbutil::gettime() - t;
		std::cerr << "Time taken: " << t << "s" << endl;
}

// Populate pointsX, pointsY and angles
void calculcate_points(
		float* pointsX, float* pointsY,
		short* angles,
		const float* sin_map, const float* cos_map,
		float line_length,
		const float length_decrease,
		const int rotation_angle_degrees,
		const unsigned int no_of_points
)
{
		// i and i+1 are the indices of the current point being edited
		// j is the index of the line/point on which they both depend
		// /p2_Test is used so that when this is a power of 2, we decrease the length
		for(unsigned int i = 1, j=0, p2_test=2; i < no_of_points; i+=2, ++j, ++p2_test){
				unsigned int i2 = i+1;

				// one line will rotate one way and the other will rotate the otehr direction
				angles[i] = angles[j] + rotation_angle_degrees;
				angles[i2] = angles[j] - rotation_angle_degrees;
				// Loop the angles around if necessary
				if(angles[i] >= 180)
						angles[i] -= 360;
				if(angles[i2] < -180)
						angles[i2] += 360;

				// Calculate the new points position
				pointsX[i] = pointsX[j] + line_length * sin_map[ angles[i]+180 ];
				pointsY[i] = pointsY[j] + line_length * cos_map[ angles[i]+180 ];

				pointsX[i2] = pointsX[j] + line_length * sin_map[ angles[i2]+180 ];
				pointsY[i2] = pointsY[j] + line_length * cos_map[ angles[i2]+180 ];

				if((p2_test & (p2_test - 1)) == 0)
						line_length *= length_decrease;
		}
}
