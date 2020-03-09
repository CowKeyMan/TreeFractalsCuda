#include "shared/workingWithDegrees.h"
#include "shared/jbutil.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <map>
#include <climits>

using std::cout;
using std::endl;
using std::string;

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

void matrix_fill_default(matrix<int> &m, int init_value);

void map_points_to_pixels(
		const int image_width, const int image_height,
		const int no_of_points,
		float* pointsX, float* pointsY,
		const float maxX, const float minY, const float maxY
);

void draw_lines(
		const unsigned long long no_of_points,
		float* pointsX, float* pointsY,
		matrix<int> &m_image
);

void draw_line(
		float pointX1, float pointY1,
		float pointX2, float pointY2,
		matrix<int> &m_image
);

// Inputs:
//    int image_width - in pixels
//    int image_height - in pixels
//				int rotation_angle_degrees (The amount to rotate per iteration [must be prefectly divisible by 8, 0 < R < 180])
//				float initial_length (length of first line)
//    float length_decrease (percentge of next line in iteration)
//    iterations (number of iterations)
int main(){
		int image_width = 512*8, image_height=512*8;
		string outfile = "out.pgm";
		float initial_length = 50; // Initial length of first line
		float length_decrease = 1.1; // How much line gets smaller per iteration
		int rotation_angle_degrees = 10; // How much to turn  per iteration (in degrees)
		int iterations = 15; // Must be between 1 and 26 (both included) otherwise it uses too much memory, as memory usage is 10x2^iterations bytes

		// get the number of points we will have by the end
		unsigned long long no_of_points = (1 << iterations); // 2^iterations

		matrix<int> m_image;
		m_image.resize(image_height, image_width);

		// lists of the properties belonging to each point
		short* angles = (short*)malloc(no_of_points * sizeof(short));
		angles[0] = 0;
		angles[1] = 0;
		float* pointsX = (float*)malloc(no_of_points * sizeof(float));
		pointsX[0] = 0;
		pointsX[1] = 0;
		float* pointsY = (float*)malloc(no_of_points * sizeof(float));
		pointsY[0] = 0;
		pointsY[1] = initial_length;

		double t = jbutil::gettime();

		// Make image all white
		matrix_fill_default(m_image, 255);
		//
		// call to use the sin and cosine maps defined in workingWithDegrees.h
		populate_sin_map(sin_map);
		populate_cos_map(cos_map);

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

		draw_lines(
				no_of_points,
				pointsX, pointsY,
				m_image
		);

		t = jbutil::gettime() - t;

		image<int> image_out = image<int>(image_height, image_width, 1, 255);
		image_out.set_channel(0, m_image);

		// save image
		std::ofstream file_out(outfile.c_str());
		image_out.save(file_out);

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
		for(unsigned long long i=2, i2=3, j=1, p2_test=2; i < no_of_points; i+=2, i2+=2, ++j, ++p2_test){
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

				if((p2_test & (p2_test - 1)) == 0) // check if p2_test is power of 2
						line_length *= length_decrease;
		}
}

void matrix_fill_default(matrix<int> &m, int init_value){
		int rows = m.get_rows();
		int cols = m.get_cols();
		for(int r = 0; r < rows; ++r){
				for(int c = 0; c < cols; ++c){
						m(r, c) = init_value;
				}
		}
}


void map_points_to_pixels(
		int image_width, int image_height,
		const int no_of_points,
		float* pointsX, float* pointsY,
		const float maxX, const float minY, const float maxY
)
{
		--image_height;
		--image_width;
		float x_mul = image_width/(maxX*2);
		float x_add = image_width/2.0f;
		float y_mul = - image_height/(maxY-minY);
		float y_add = image_height - image_height/(maxY-minY) * -minY; // The 0 coordinate (remember we want to switch the y coordiante upside down
		
		for(int i = 0; i < no_of_points; ++i){
				pointsX[i] = pointsX[i] * x_mul + x_add;
				pointsY[i] = pointsY[i] * y_mul + y_add;
		}
}

void draw_lines(
		const unsigned long long no_of_points,
		float* pointsX, float* pointsY,
		matrix<int> &m_image
)
{
		draw_line(pointsX[0], pointsY[0], pointsX[1], pointsY[1], m_image);
		for(unsigned long long i=2, i2=3, j=1; i < no_of_points; i+=2, i2+=2, ++j){
				draw_line(pointsX[i], pointsY[i], pointsX[j], pointsY[j], m_image);
				draw_line(pointsX[i2], pointsY[i2], pointsX[j], pointsY[j], m_image);
		}
}

void draw_line(
		float pointX1, float pointY1,
		float pointX2, float pointY2,
		matrix<int> &m_image
)
{
		bool x1_greaterThan_2 = pointX1 > pointX2;
		bool y1_greaterThan_2 = pointY1 > pointY2;

		float differenceX = (x1_greaterThan_2)? pointX1-pointX2 : pointX2-pointX1;
		float differenceY = (y1_greaterThan_2)? pointY1-pointY2 : pointY2-pointY1;

		if(differenceX > differenceY){ // move by x
				if(x1_greaterThan_2){ // we need to go from point 2 -> 1
						float y_position = pointY2; // start at Y point 2
						int startX = round(pointX2);
						int endX = round(pointX1);
						float moveY = (pointY1-pointY2)/differenceX;
				
						for(int i = startX; i < endX; ++i, y_position+=moveY){
								m_image(round(y_position), i) = 0;
						}
				}
				else
				{
						float y_position = pointY1; // start at Y point 1
						int startX = round(pointX1);
						int endX = round(pointX2);
						float moveY = (pointY2-pointY1)/differenceX;
					
						for(int i = startX; i < endX; ++i, y_position+=moveY){
								m_image(round(y_position), i) = 0;
						}
				}
		}
		else
		{
				if(y1_greaterThan_2){ // we need to go from point 2 -> 1
						float x_position = pointX2; // start at Y point 2
						int startY = round(pointY2);
						int endY = round(pointY1);
						float moveX = (pointX1-pointX2)/differenceY;

						for(int i = startY; i < endY; ++i, x_position+=moveX){
								m_image(i, round(x_position)) = 0;
						}
				}
				else
				{
						float x_position = pointX1; // start at Y point 1
						int startY = round(pointY1);
						int endY = round(pointY2);
						float moveX = (pointX2-pointX1)/differenceY;

						for(int i = startY; i < endY; ++i, x_position+=moveX){
								m_image(i, round(x_position)) = 0;
						}
				}
		}
}
