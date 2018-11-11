/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <cassert>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "ssrc/spatial/kd_tree.h"

using namespace std;

ParticleFilter::ParticleFilter()
    : num_particles(1000), is_initialized(false)
    { }

void ParticleFilter::init(double x, double y, double theta, const std::array<double, 3>& std_pos) {
    // This assertion seems pointless but ensures that we don't accidentally
    // change the dimensionality of the array and forget about adjustments here.
    assert(std_pos.size() == 3);

    // We make sure we don't add more particles than expected.
    particles.clear();
    weights.clear();

    // By setting the mean of the random noise distribution to the specified GPS coordinates,
    // we create random particles centered around our GPS position estimate.
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    // Initially, all particles are equally likely, so we set their weight to an identical value.
    const auto initial_weight = 1.0;

    // We now initialize all particles.
    assert(this->num_particles > 0);
    const auto num_particles = this->num_particles;
    for (auto p = 0U; p < num_particles; ++p) {
        const Particle particle_temp{
                .id = p,
                .x = dist_x(gen),
                .y = dist_y(gen),
                .theta = dist_theta(gen),
                .weight = initial_weight
        };
        particles.push_back(particle_temp);
        weights.push_back(initial_weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, const std::array<double, 3>& std_pos, double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.

    default_random_engine gen;

    const auto nonzero_yaw = fabs(yaw_rate) > 0;
    const auto yaw_rate_dt = yaw_rate * delta_t;
    const auto velocity_dt = velocity * delta_t;
    const auto velocity_over_yaw_rate = nonzero_yaw ? velocity/yaw_rate : 0;

    for (auto &particle : particles) {
        auto x = particle.x;
        auto y = particle.y;
        auto theta = particle.theta;

        // Update the pose.
        if (nonzero_yaw) {
            x += velocity_over_yaw_rate * (sin(theta + yaw_rate_dt) - sin(theta));
            y += velocity_over_yaw_rate * (cos(theta) - cos(theta + yaw_rate_dt));
            theta += yaw_rate_dt;
        }
        else {
            x += velocity_dt * cos(theta);
            y += velocity_dt * sin(theta);
        }

        // We add a small gaussian noise to each (updated) particle by resampling around the
        // updated position.
        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);

        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
#ifdef USE_ASSOCIATION_NAIVE
    return dataAssociationNaive(predicted, observations);
#else // USE_ASSOCIATION_NAIVE
    return dataAssociationTree(predicted, observations);
#endif // USE_ASSOCIATION_NAIVE
}

void ParticleFilter::dataAssociationTree(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
    typedef std::array<double, 2> Point;
    typedef ssrc::spatial::kd_tree<Point, size_t> Tree;

    Tree tree;
    for (auto l = 0U; l < predicted.size(); ++l) {
        const auto landmark = predicted[l];
        Point pt { landmark.x, landmark.y };
        tree[pt] = l;
    }

    for (auto &observation : observations) {
        Point query { observation.x, observation.y };
        auto range = tree.find_nearest_neighbors(query, 1);
        for (Tree::knn_iterator it = range.first; it != range.second; ++it) {
            std::pair<Point, size_t> p = *it;
            observation.landmark_id = p.second;

            // There is only one match.
            // TODO: Use find_nearest_neighbor() (instead of _neighbors())
            break;
        }
    }
}

void ParticleFilter::dataAssociationNaive(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
    // To find the predicted measurement that is closest to each observed measurement,
    // we're going to run a brute force comparison between all observations and predicted observations.
    // A naive implementation like this gets the job done, but is terrible inefficient for large number of landmarks.
    // What helps us here is that we only look through all landmarks actually in sensor range.
    //
    // An alternative solution would be to sort predicted observations into a quadtree in order to quickly
    // obtain candidate answers, then run a brute force match on the candidates.
    // We'll leave that as an optimization for a later iteration.
    // TODO: Optimize k-NN match performance
    const auto max_dist = std::numeric_limits<double>::max();
    const auto num_predictions = predicted.size();

    // TODO: Parallelize outer loop using OpenMP for a simple performance improvement.
    for (auto &observation : observations) {
        double min_dist = max_dist;
        for (auto l = 0U; l < num_predictions; ++l) {
            const auto& landmark = predicted[l];
            // For distance comparisons we don't need the Euclidean distance (which involves a square root)
            // but can make use of the squared Euclidean distance. Since to the square root function is monotonic,
            // results would be the same for comparison purposes.
            const auto distance = dist_sq(landmark.x, landmark.y, observation.x, observation.y);
            if (min_dist < distance) continue;

            // Update the best match. Note that we're using the loop index as the
            // landmark ID to ensure we pick a valid index from the range of predictions, rather
            // then from the range of original landmarks.
            min_dist = distance;
            observation.landmark_id = l;
        }
    }
}

vector<LandmarkObs> ParticleFilter::getLandmarksInRange(double sensor_range, const Map &map_landmarks, const Particle& particle) const {
    // Rather than calculating the Euclidean distance, we're going to use the squared Euclidean distance again.
    // For the comparison to work, we'll also use the squared sensor range.
    const auto sensor_range_sq = sensor_range * sensor_range;

    vector<LandmarkObs> landmarks_in_range;
    for (const auto& map_landmark : map_landmarks.landmark_list) {
        const auto landmark_dist_sq = dist_sq(particle.x, particle.y, map_landmark.x_f, map_landmark.y_f);
        if (landmark_dist_sq > sensor_range_sq) continue;

        landmarks_in_range.push_back({
                                             .landmark_id = map_landmark.id_i,
                                             .x = map_landmark.x_f,
                                             .y = map_landmark.y_f
                                     });
    }
    return landmarks_in_range;
}

vector<LandmarkObs> ParticleFilter::transformVehicleToMapSpace(const Particle& particle, const vector<LandmarkObs>& observations) const {
    const auto sin_theta = sin(particle.theta);
    const auto cos_theta = cos(particle.theta);
    const auto px = particle.x;
    const auto py = particle.y;

    vector<LandmarkObs> observations_transformed;
    for (const auto& observation : observations) {
        const auto ox = observation.x;
        const auto oy = observation.y;
        const auto landmark_id = 0U; // observations will be assigned in a later step
        observations_transformed.push_back({
            .landmark_id = landmark_id,
            .x = px + cos_theta * ox - sin_theta * oy,
            .y = py + sin_theta * ox + cos_theta * oy
        });
    }
    return observations_transformed;
}

void ParticleFilter::updateWeights(double sensor_range, const std::array<double, 2>& std_landmark,
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // This assertion seems pointless but ensures that we don't accidentally
    // change the dimensionality of the array and forget about adjustments here.
    assert(std_landmark.size() == 2);
    const auto sigma_x = std_landmark[0];
    const auto sigma_y = std_landmark[1];
    assert(sigma_x >= 0);
    assert(sigma_y >= 0);

    // Pre-calculate some coefficients.
    const double gauss_norm = 1.0 / (2.0 * M_PI * sigma_x * sigma_y);
    const double one_over_two_sig_x_sq = 1.0 / (2.0 * square(sigma_x));
    const double one_over_two_sig_y_sq = 1.0 / (2.0 * square(sigma_y));

    // Remove old weights, since we're going to determine new ones.
    weights.clear();

    for (auto &particle : particles) {
        // For further processing, keep only those landmarks actually within sensor range.
        const auto landmarks_in_range = getLandmarksInRange(sensor_range, map_landmarks, particle);

        // Nothing to do if there are no observations within sensor range.
        if (landmarks_in_range.empty()) {
            return;
        }

        // Transform observations from vehicle to map space.
        auto observations_transformed = transformVehicleToMapSpace(particle, observations);

        // Assign each observation to a possible landmark.
        // TODO: By utilizing sensor covariance, we might make use of the Mahalanobis distance.
        dataAssociation(landmarks_in_range, observations_transformed);

        // Calculate the particle weight and set associations.
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;

        particle.weight = 1;
        for (const auto& observation : observations_transformed) {
            const auto obs_x = observation.x;
            const auto obs_y = observation.y;

            // We need to make sure the associated landmark is actually in the
            // list of pre-selected landmarks.
            assert(observation.landmark_id < landmarks_in_range.size());

            // Determine the closest landmark.
            const auto closest_landmark = landmarks_in_range[observation.landmark_id];
            const auto landmark_x = closest_landmark.x; // mu_x
            const auto landmark_y = closest_landmark.y; // mu_y

            // Determine the weight.
            const auto exponent = square(obs_x - landmark_x) * one_over_two_sig_x_sq
                                + square(obs_y - landmark_y) * one_over_two_sig_y_sq;
            const auto weight = gauss_norm * exp(-exponent);
            assert(weight >= 0.0);

            // Update the particle.
            particle.weight *= weight;
            associations.push_back(closest_landmark.landmark_id);
            sense_x.push_back(static_cast<double>(closest_landmark.x));
            sense_y.push_back(static_cast<double>(closest_landmark.y));
        }

        assert(particle.weight >= 0.0);
        setAssociations(particle, associations, sense_x, sense_y);
        weights.push_back(particle.weight);
    }
}

void ParticleFilter::resample() {
    // Prepare the new set of particles.
    vector<Particle> resampled;
    resampled.reserve(particles.size());

    // Build the particle weights vector.
    vector<double> weights;
    weights.reserve(particles.size());
    for(const auto& p : particles){
        weights.push_back(p.weight);
    }

    // Sample with replacement with a probability proportional to the weights.
    std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    for(auto i = 0U; i < weights.size(); ++i){
        const auto sampled_idx = d(gen);
        resampled.push_back(particles[sampled_idx]);
    }

    particles = resampled;
}

Particle& ParticleFilter::setAssociations(Particle &particle, const std::vector<int> &associations,
                                          const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(const Particle& best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(const Particle& best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(const Particle& best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
