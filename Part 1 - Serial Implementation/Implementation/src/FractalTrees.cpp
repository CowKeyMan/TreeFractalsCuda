#include "shared/jbutil.h"
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

using jbutil::matrix;
using jbutil::image;

float degrees_to_radians(int degrees);
void populate_sin_map(float* sin_map);
void populate_cos_map(float* cos_map);

void calculcate_points(
		float* pointsX, float* pointsY,
		short* angles,
		const float* sin_map, const float* cos_map,
		const float line_length, const float length_multiplier,
		const int rotation_angle_degrees,
		const unsigned long no_of_points,
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
		const unsigned long no_of_points,
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
//    float length_multiplier (multiply the current line's length by this number
//	  int rotation_angle_degrees (The amount to rotate per iteration) must be between 0 and 180
//    iterations (number of iterations) Must be between 1 and 26 (both included) otherwise it uses too much memory, as memory usage is 10x2^iterations bytes
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
		//For testing
		/*
		int image_width = 512, image_height=512;
		float length_multiplier = 1;
		int rotation_angle_degrees = 20;
		int iterations = 12;
		*/

		float initial_length = 1;
		// get the number of points we will have by the end
		unsigned long no_of_points = (2 << (iterations-1)); // 2^iterations

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
		

		float sin_map[360]; // maps angle i to its sin at index i+180 (-180 <= i <= 180)
		float cos_map[360]; // maps angle i to its cos at index i+180 (-180 <= i <= 180)

		float maxX, minY, maxY;

		matrix<int> m_image;
		m_image.resize(image_height, image_width);

		// --------- START TIMING PART 1 ----------
		double t = jbutil::gettime();

		populate_sin_map(sin_map);
		populate_cos_map(cos_map);

		calculcate_points(
				pointsX, pointsY,
				angles,
				sin_map, cos_map,
				initial_length * length_multiplier,
				length_multiplier,
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

		// --------- STOP TIMING PART 1 ----------
		t = jbutil::gettime() - t;
		std::cerr << "Time taken to generate the points: " << t << "s" << endl;

		// --------- START TIMING PART 2 ----------
		t = jbutil::gettime();

		matrix_fill_default(m_image, 255); // Initialise to all white

		draw_lines(
				no_of_points,
				pointsX, pointsY,
				m_image
		);

		// --------- STOP TIMING PART 2 ----------
		t = jbutil::gettime() - t;
		std::cerr << "Time taken to create the image: " << t << "s" << endl;

		image<int> image_out = image<int>(image_height, image_width, 1, 255);
		image_out.set_channel(0, m_image);

		// save image
		char outfile[150];
		sprintf(outfile, "output_images/%d x%d _m=%.2f _theta=%d _n= %.0d.pgm",
				image_width, image_height,
				length_multiplier,
				rotation_angle_degrees,
				iterations);
		std::ofstream file_out(outfile);
		image_out.save(file_out);
}

float degrees_to_radians(int degrees){
		return degrees * pi / 180;
}

void populate_sin_map(float* sin_map){
		assert(sin_map != NULL);
		for(int i = -180; i < 179; i++){
				sin_map[i+180] = sin(degrees_to_radians(i));
		}
}

void populate_cos_map(float* cos_map){
		assert(cos_map != NULL);
		for(int i = -180; i < 179; i++){
				cos_map[i+180] = cos(degrees_to_radians(i));
		}
}

// Populate pointsX, pointsY and angles
void calculcate_points(
		float* pointsX, float* pointsY,
		short* angles,
		const float* sin_map, const float* cos_map,
		float line_length,
		const float length_multiplier,
		const int rotation_angle_degrees,
		const unsigned long no_of_points,
		float &maxX, float &minY, float &maxY
)
{
		assert(pointsX != NULL && pointsY != NULL && angles != NULL && sin_map != NULL && cos_map != NULL);

		// i and i+1 are the indices of the current point being edited
		// j is the index of the line/point on which they both depend
		// /p2_Test is used so that when this is a power of 2, we decrease the length
		for(unsigned long i=2, i2=3, j=1, p2_test=2; i < no_of_points; i+=2, i2+=2, ++j, ++p2_test){
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
						line_length *= length_multiplier;
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
		assert(pointsX != NULL && pointsY != NULL);

		--image_height;
		--image_width;
		float x_mul = (maxX == 0)? 1: image_width/(maxX*2);
		float x_add = image_width/2.0f;
		float y_mul = (maxY==minY)? image_height : -image_height/(maxY-minY);
		float y_add = (maxY==minY)? 0 : image_height + image_height/(maxY-minY) * minY; // The 0 coordinate (remember we want to switch the y coordiante upside down

		for(int i = 0; i < no_of_points; ++i){
				pointsX[i] = pointsX[i] * x_mul + x_add;
				pointsY[i] = pointsY[i] * y_mul + y_add;

				assert(pointsX[i] > -1 && pointsX[i] < image_width+1 && pointsY[i] > -1 && pointsY[i] < image_height+1);
		}
}

void draw_lines(
		const unsigned long no_of_points,
		float* pointsX, float* pointsY,
		matrix<int> &m_image
)
{
		assert(pointsX != NULL && pointsY != NULL);

		draw_line(pointsX[0], pointsY[0], pointsX[1], pointsY[1], m_image);
		for(unsigned long i=2, i2=3, j=1; i < no_of_points; i+=2, i2+=2, ++j){
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
