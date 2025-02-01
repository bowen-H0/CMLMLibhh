#ifndef CMLMLibhh_HPP
#define CMLMLibhh_HPP
#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <type_traits>
#include <sstream>

namespace CMLMLibhh
{
    // Helper function: Convert a vector of integers to a space-separated string
    inline std::string vectorToString(const std::vector<int> &vec)
    {
        std::ostringstream oss;
        for (const auto &v : vec)
        {
            oss << v << " ";
        }
        return oss.str();
    }

    // Define the Q_learning class, supporting multi-dimensional state spaces and multiple template types
    template <typename T>
    class Q_learning
    {
    private:
        std::mt19937 rng{std::random_device{}()}; // Random number generator
        std::uniform_real_distribution<> dist{0.0, 1.0}; // Uniform distribution between 0 and 1

        // Helper function: Compute the one-dimensional state index
        int getStateIndex(const std::vector<int> &state);

    public:
        int status_dimension = 1;      // State dimension
        std::vector<int> actions;      // Action set
        std::vector<int> dimensions;   // Size of each dimension
        std::vector<std::vector<T>> Q; // Q-table
        float alpha = 0.4;             // Learning rate
        float gamma = 0.9;             // Discount factor
        float greedy_prob = 0.70;      // Probability for greedy strategy

        // Initialization function
        bool init(const std::vector<int> &dim, const std::vector<int> &actionSet);

        // Get the Q-value for a given state and action
        T getQValue(const std::vector<int> &state, int action);

        // Set the Q-value
        void setQValue(const std::vector<int> &state, int action, T value);

        // Choose an action
        int chooseAction(const std::vector<int> &state);

        // Update the Q-value
        void updateQ(const std::vector<int> &state, int action, const std::vector<int> &nextState, T reward);
    };

} // namespace CMLMLibhh

#endif // CMLMLibhh_HPP
