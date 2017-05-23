/*
 * particle_filter.cpp
 *
 *  Created on: May 12, 2017
 *  Author: Alex Cui
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x, y, psi
	normal_distribution<double> dist_x_init(x, std[0]);
	normal_distribution<double> dist_y_init(y, std[1]);
	normal_distribution<double> dist_theta_init(theta, std[2]);

	num_particles = 100;

	weights.resize(num_particles);

	for (int i = 0; i < num_particles; ++i) {
		Particle p = {
			i,
			dist_x_init(gen),
			dist_y_init(gen),
			dist_theta_init(gen),
			1.0
		};
		particles.push_back(p);
	}

	// finish initialization
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for (auto &p: particles) {
		if (fabs(yaw_rate) > 0.001) {
			p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
        	p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
        	p.theta = p.theta + delta_t * yaw_rate;
		} else {
			p.x = p.x + velocity * delta_t * cos(p.theta);
            p.y = p.y + velocity * delta_t * sin(p.theta);

		}

		// add gaussian noise
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_theta(p.theta, std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);

	}
}

// Combine it in the next function
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for(int i=0;i<observations.size(); ++i)
    {
         LandmarkObs  obs_land = observations[i];
         double lowest_dist, new_x,new_y;
         for(int j=0;j<predicted.size(); ++j)
         {
             LandmarkObs pred_land = predicted[j];
             double distance = dist(pred_land.x,pred_land.y, obs_land.x,obs_land.y);
             switch(j)
             {
                 case 0:{
                        new_x = pred_land.x;
                        new_y = pred_land.y;
                        lowest_dist=distance;
                     break;
                 }
                 default:
                 {
                     if(distance<lowest_dist){
                          new_x = pred_land.x;
                          new_y = pred_land.y;
                          lowest_dist=distance;

                      }
                     break;
                 }
              }
          }//end inner for


     observations[i].y= new_y;
     observations[i].x= new_x;

    }

    return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i=0; i<num_particles; ++i) {
		
		Particle &p = particles[i];
		
		// init the weight
		double weight = 1.0;

		for (auto &obs: observations) {
			
			// http://planning.cs.uiuc.edu/node99.html
			// convert to map coordinates

			double obs_x, obs_y;
            obs_x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            obs_y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);

            // assign each observation to the closest landmark
            Map::single_landmark_s closest_landmark = {0, 0.0, 0.0};
            double min_dist_obs_to_landmark = sensor_range;
            
            for (auto &landmark: map_landmarks.landmark_list) {
                
                // calculate particle to landmark distrance

            	double distance_particle_to_landmark = dist(p.x, p.y, landmark.x_f, landmark.y_f);

            	if (distance_particle_to_landmark <= sensor_range) {
            		double distance_obs_to_landmark = dist(obs_x, obs_y, landmark.x_f, landmark.y_f);
            		if (distance_obs_to_landmark < min_dist_obs_to_landmark) {
            			min_dist_obs_to_landmark =  distance_obs_to_landmark;
            			closest_landmark = landmark;
            		}
            	}
            }

            double x_diff = closest_landmark.x_f - obs_x;
            double y_diff = closest_landmark.y_f - obs_y;
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];

            double x_y_term = ((x_diff*x_diff)/(std_x*std_x)) + ((y_diff*y_diff)/(std_y*std_y));
            long double w = exp(-0.5*x_y_term) / (2*M_PI*std_x*std_y);

            weight = weight * w;

		}

		p.weight = weight;
		weights[i] = weight;
	}

	// weights normalization
	double weights_sum = accumulate(weights.begin(), weights.end(), 0.0);

	for (int i =0; i<num_particles; ++i) {
		particles[i].weight = particles[i].weight / weights_sum;
		weights[i] = particles[i].weight;
	}


}

void ParticleFilter::resample() {
	
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;

	// will resamples according to the weights

	discrete_distribution<> dd(weights.begin(), weights.end());
	vector<Particle> resampled_particles;

	for (int i=0; i<num_particles; ++i) {
		int resampled_particle_index = dd(gen);
		resampled_particles.push_back(particles[resampled_particle_index]);
	}

	particles = resampled_particles;
	
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
