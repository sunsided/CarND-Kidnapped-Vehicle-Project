#include <array>
#include <limits>
#include "particle_filter.h"

using namespace std;

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
