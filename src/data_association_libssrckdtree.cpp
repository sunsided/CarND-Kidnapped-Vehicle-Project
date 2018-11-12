#include <array>
#include "ssrc/spatial/kd_tree.h"
#include "particle_filter.h"

using namespace std;
using namespace ssrc;

#if !ASSOCIATION_NAIVE

void ParticleFilter::dataAssociationTree(const std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
    typedef std::array<double, 2> Point;
    typedef spatial::kd_tree<Point, size_t> Tree;

    Tree tree{};
    for (auto l = 0U; l < predicted.size(); ++l) {
        const auto landmark = predicted[l];
        const Point pt { landmark.x, landmark.y };
        tree[pt] = l;
    }

    for (auto &observation : observations) {
        const Point query { observation.x, observation.y };
        auto match = tree.find_nearest_neighbor(query, 1);
        std::pair<Point, size_t> p = *match;
        observation.landmark_id = p.second;
    }
}

#endif