// --------------------------------------------------------------------------------
/**
	localizer.cpp

	Purpose: implements a 2-dimensional histogram filter
	for a robot living on a colored cyclical grid by 
	correctly implementing the "initialize_beliefs", 
	"sense", and "move" functions.

  I have done modify code here
*/

// --------------------------------------------------------------------------------
#include "helpers.cpp"
#include <stdlib.h>
#include "debugging_helpers.cpp"

// --------------------------------------------------------------------------------
using namespace std;

// --------------------------------------------------------------------------------
/**
	TODO - implement this function --> done
    
    Initializes a grid of beliefs to a uniform distribution. 

    @param grid - a two dimensional grid map (vector of vectors 
    	   of chars) representing the robot's world. For example:
    	   
    	   g g g
    	   g r g
    	   g g g
		   
		   would be a 3x3 world where every cell is green except 
		   for the center, which is red.

    @return - a normalized two dimensional grid of floats. For 
           a 2x2 grid, for example, this would be:

           0.25 0.25
           0.25 0.25
*/
vector< vector <float> > initialize_beliefs(vector< vector <char> > grid) 
{
  // beliefs vector matrix
	vector< vector <float> > beliefs;

  // check dimension
	int height = grid.size();
  int width = grid[0].size();
  int area = height * width;
  float belief_per_cell = 1.0 / area;

  // zeroing
  beliefs = zeros(height, width);
  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      beliefs[i][j] = belief_per_cell;
    }
  }  

  // return init result
  return beliefs;
}

// --------------------------------------------------------------------------------
/**
	TODO - implement this function --> done 
    
    Implements robot sensing by updating beliefs based on the 
    color of a sensor measurement 

	@param color - the color the robot has sensed at its location

	@param grid - the current map of the world, stored as a grid
		   (vector of vectors of chars) where each char represents a 
		   color. For example:

		   g g g
    	   g r g
    	   g g g

   	@param beliefs - a two dimensional grid of floats representing
   		   the robot's beliefs for each cell before sensing. For 
   		   example, a robot which has almost certainly localized 
   		   itself in a 2D world might have the following beliefs:

   		   0.01 0.98
   		   0.00 0.01

    @param p_hit - the RELATIVE probability that any "sense" is 
    	   correct. The ratio of p_hit / p_miss indicates how many
    	   times MORE likely it is to have a correct "sense" than
    	   an incorrect one.

   	@param p_miss - the RELATIVE probability that any "sense" is 
    	   incorrect. The ratio of p_hit / p_miss indicates how many
    	   times MORE likely it is to have a correct "sense" than
    	   an incorrect one.

    @return - a normalized two dimensional grid of floats 
    	   representing the updated beliefs for the robot. 
*/
vector< vector <float> > sense(char color, 
                               vector< vector <char> > grid, 
                               vector< vector <float> > beliefs, 
                               float p_hit,
                               float p_miss) 
{
  // new_beliefs vector matrix
	vector< vector <float> > new_beliefs;

  // check dimension
	int height = grid.size();
  int width = grid[0].size();
  int area = height * width;
  float belief_per_cell = 1.0 / area;

  // zeroing
  new_beliefs = zeros(height, width);
  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      new_beliefs[i][j] = belief_per_cell;
    }
  }

  // sense
  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      if (color == grid[i][j]) 
      {
        new_beliefs[i][j] *= p_hit; 
      }
      else 
      {
        new_beliefs[i][j] *= p_miss;
      }
    }
  }

  // return sense result
  return normalize(new_beliefs);
}


// --------------------------------------------------------------------------------
/**
	TODO - implement this function --> done 
    
    Implements robot motion by updating beliefs based on the 
    intended dx and dy of the robot. 

    For example, if a localized robot with the following beliefs

    0.00  0.00  0.00
    0.00  1.00  0.00
    0.00  0.00  0.00 

    and dx and dy are both 1 and blurring is 0 (noiseless motion),
    than after calling this function the returned beliefs would be

    0.00  0.00  0.00
    0.00  0.00  0.00
    0.00  0.00  1.00 

	@param dy - the intended change in y position of the robot

	@param dx - the intended change in x position of the robot

   	@param beliefs - a two dimensional grid of floats representing
   		   the robot's beliefs for each cell before sensing. For 
   		   example, a robot which has almost certainly localized 
   		   itself in a 2D world might have the following beliefs:

   		   0.01 0.98
   		   0.00 0.01

    @param blurring - A number representing how noisy robot motion
           is. If blurring = 0.0 then motion is noiseless.

    @return - a normalized two dimensional grid of floats 
    	   representing the updated beliefs for the robot. 
*/
vector< vector <float> > move(int dy, int dx, 
                              vector < vector <float> > beliefs,
                              float blurring) 
{
  // new_G vector matrix
	vector < vector <float> > new_G;

  // variables of height and width
	int height = beliefs.size();
  int width = beliefs[0].size();

  // zeroing new_G matrix
  new_G = zeros(height, width);

  // move process
  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      int new_i = (i + dy + height) % height;
      int new_j = (j + dx + width) % width;
      new_G[new_i][new_j] = beliefs[i][j];
    }
  }

  // return blur result
  return blur(new_G, blurring);
}

// --------------------------------------------------------------------------------