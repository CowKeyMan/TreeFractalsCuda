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
		const unsigned long long no_of_points,
		float &maxX, float &minY, float &maxY
);

void map_points_to_pixels(
		const int image_width, const int image_height,
		const int no_of_points,
		float* pointsX, float* pointsY,
		const float maxX, const float minY, const float maxY
);

void drawLines(
		const unsigned long long no_of_points,
		float* pointsX, float* pointsY,
		matrix<int> image
);


// Inputs:
//    int image_width - in pixels
//    int image_height - in pixels
//				int rotation_angle_degrees (The amount to rotate per iteration [must be prefectly divisible by 8, 0 < R < 180])
//				float initial_length (length of first line)
//    float length_decrease (percentge of next line in iteration)
//    iterations (number of iterations)
int main(){
		int image_width = 512, image_height=512;
		float initial_length = 50; // Initial length of first line
		float length_decrease = 0.9f; // How much line gets smaller per iteration
		int rotation_angle_degrees = 5; // How much to turn  per iteration (in degrees)
		int iterations = 26; // Must be between 1 and 26 (both included) otherwise it uses too much memory, as memory usage is 10x2^iterations bytes

		// get the number of points we will have by the end
		unsigned long long no_of_points = (1 << iterations) - 1; // 2^iterations - 1

		cout << no_of_points << endl;

		// lists of the properties belonging to each point
		short* angles = (short*)malloc(no_of_points * sizeof(short));
		angles[0] = 0;
		float* pointsX = (float*)malloc(no_of_points * sizeof(float));
		pointsX[0] = 0;
		float* pointsY = (float*)malloc(no_of_points * sizeof(float));
		pointsY[0] = initial_length;

		// call to use the sin and cosine maps defined in workingWithDegrees.h
		populate_sin_map(sin_map);
		populate_cos_map(cos_map);
		
		double t = jbutil::gettime();

		float maxX, minY, maxY;

		calculcate_points(
				pointsX, pointsY,
				angles,
				sin_map, cos_map,
				initial_length,
				length_decrease,
				rotation_angle_degrees,
				no_of_points,
				maxX, minY, maxY
		);

		free(angles);

		map_points_to_pixels(
				image_width, image_height,
				no_of_points,
				pointsX, pointsY,
				maxX, minY, maxY
		);


		t = jbutil::gettime() - t;
		cout << "(" << pointsX[no_of_points-1] << ", " <<  pointsY[no_of_points-1] << ")" << endl;
		cout << maxX << " " << minY << " " << maxY << endl;
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
		const unsigned long long no_of_points,
		float &maxX, float &minY, float &maxY
)
{
		// i and i+1 are the indices of the current point being edited
		// j is the index of the line/point on which they both depend
		// /p2_Test is used so that when this is a power of 2, we decrease the length
		for(unsigned long long i = 1, j=0, p2_test=2; i < no_of_points; i+=2, ++j, ++p2_test){
				unsigned long long i2 = i+1;

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
				if (pointsX[i] > maxX)
						maxX =		pointsX[i];
				pointsY[i] = pointsY[j] + line_length * cos_map[ angles[i]+180 ];
				if (pointsY[i] > maxY)
						maxY =		pointsY[i];
				if (pointsY[i] < minY)
						minY = pointsY[i];

				pointsX[i2] = pointsX[j] + line_length * sin_map[ angles[i2]+180 ];
				if (pointsX[i2] > maxX)
						maxX =		pointsX[i2];
				pointsY[i2] = pointsY[j] + line_length * cos_map[ angles[i2]+180 ];
				if (pointsY[i2] > maxY)
						maxY =		pointsY[i2];
				if (pointsY[i2] < minY)
						minY = pointsY[i2];

				if((p2_test & (p2_test - 1)) == 0)
						line_length *= length_decrease;
		}
}

void map_points_to_pixels(
		const int image_width, const int image_height,
		const int no_of_points,
		float* pointsX, float* pointsY,
		const float maxX, const float minY, const float maxY
)
{
		float x_mul = image_width/(maxX*2);
		float x_add = image_width/2;
		float y_mul = - image_height/(maxY-minY);
		float y_add = image_height - (maxY-minY)/maxY * image_height; // The 0 coordinate (remember we want to switch the y coordiante upside down
		
		for(int i = 0; i < no_of_points; ++i){
				pointsX[i] = pointsX[i] * x_mul + x_add;
				pointsY[i] = pointsY[i] * y_mul + y_add;
		}
}

void drawLines(
		const unsigned long long no_of_points,
		float* pointsX, float* pointsY,
		matrix<int> image
)
{
		
}
