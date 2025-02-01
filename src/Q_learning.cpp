#include "../include/CMLMLibhh.hpp"

// Implementation of the Q_learning class template, where T is a template type (e.g., float, double, int, etc.)
template class CMLMLibhh::Q_learning<float>;
template class CMLMLibhh::Q_learning<double>;

namespace CMLMLibhh
{

    // Helper function: Compute the one-dimensional state index
    // Converts a multi-dimensional state index into a one-dimensional array index.
    template <typename T>
    int Q_learning<T>::getStateIndex(const std::vector<int> &state)
    {
        int index = 0;      // Stores the computed state index
        int multiplier = 1; // Used to calculate the dimension weight for the state
        try
        {
            for (int i = status_dimension - 1; i >= 0; --i)
            {
                if (state[i] >= dimensions[i]) // Check if the state exceeds the boundaries
                {
                    throw std::out_of_range("State index out of bounds.");
                }
                index += state[i] * multiplier; // Compute the index value for this dimension
                multiplier *= dimensions[i];    // Update the weight
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Q_learning: Exception occurred during getStateIndex. "
                      << "State: " << CMLMLibhh::vectorToString(state)
                      << ". Error: " << e.what() << '\n';
        }

        // Compute the index starting from the last dimension

        return index; // Return the computed index
    }

    // Initialization function: Initialize state dimensions, action set, and Q-table
    template <typename T>
    bool Q_learning<T>::init(const std::vector<int> &dim, const std::vector<int> &actionSet)
    {
        if (dim.empty()) // Check if the dimensions are empty
            throw std::invalid_argument("Dimensions cannot be empty.");

        status_dimension = dim.size(); // Store the number of state dimensions
        dimensions = dim;              // Store the size of each dimension
        actions = actionSet;           // Store the action set

        int total_states = 1;
        for (int size : dimensions)
        {
            total_states *= size; // Compute the total number of possible states
        }

        Q.resize(total_states, std::vector<T>(actions.size(), T{})); // Initialize the Q-table with size [states][actions]

        return true; // Return success
    }

    // Get the Q-value for a given state and action
    template <typename T>
    T Q_learning<T>::getQValue(const std::vector<int> &state, int action)
    {
        return Q[getStateIndex(state)][action]; // Return the Q-value for the specified state and action
    }

    // Set the Q-value for a given state and action
    template <typename T>
    void Q_learning<T>::setQValue(const std::vector<int> &state, int action, T value)
    {
        Q[getStateIndex(state)][action] = value; // Update the Q-value
    }

    // Choose an action: Selects an action based on the epsilon-greedy strategy
    template <typename T>
    int Q_learning<T>::chooseAction(const std::vector<int> &state)
    {
        if (dist(rng) < greedy_prob) // Use greedy strategy
        {
            int best_action = 0;
            T maxQ = Q[getStateIndex(state)][0];
            // Find the action corresponding to the maximum Q-value
            for (int action = 1; action < actions.size(); ++action)
            {
                T q_value = Q[getStateIndex(state)][action];
                if (q_value > maxQ)
                {
                    best_action = action;
                    maxQ = q_value;
                }
            }
            return best_action; // Return the best action
        }
        else // Use random selection
        {
            std::uniform_int_distribution<int> action_dist(0, actions.size() - 1);
            return action_dist(rng); // Return a random action
        }
    }

    // Update Q-value: Updates the Q-value for a given state and action based on the Q-learning algorithm formula
    template <typename T>
    void Q_learning<T>::updateQ(const std::vector<int> &state, int action, const std::vector<int> &nextState, T reward)
    {

        int stateIndex = getStateIndex(state);         // Get the index of the current state
        int nextStateIndex = getStateIndex(nextState); // Get the index of the next state
        try
        {
            T maxQ_nextState = Q[nextStateIndex][0];
            // Find the maximum Q-value in the next state
            for (int next_action = 1; next_action < actions.size(); ++next_action)
            {
                maxQ_nextState = std::max(maxQ_nextState, Q[nextStateIndex][next_action]);
            }

            Q[stateIndex][action] += alpha * (reward + gamma * maxQ_nextState - Q[stateIndex][action]);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Q_learning: Exception occurred during Q-value update. "
                      << "State: " << CMLMLibhh::vectorToString(state)
                      << ", Action: " << action
                      << ", NextState: " << CMLMLibhh::vectorToString(nextState)
                      << ", Reward: " << reward
                      << ". Error: " << e.what() << '\n';
        }

        // Update the Q-value for the current state and action
    }

}
