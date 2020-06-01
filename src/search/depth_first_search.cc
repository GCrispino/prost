#include "depth_first_search.h"
#include "random_walk.h"

#include "prost_planner.h"

#include "utils/math_utils.h"

#include <iostream>
#include <set>
#include <logger.h>

using namespace std;

/******************************************************************
                     Search Engine Creation
******************************************************************/

DepthFirstSearch::DepthFirstSearch()
    : DeterministicSearchEngine("DFS"), rewardHelperVar(0.0) {}

/******************************************************************
                       Main Search Functions
******************************************************************/

void DepthFirstSearch::estimateQValue(State const& state, int actionIndex,
                                      double& qValue) {
    assert(state.stepsToGo() > 0);
    assert(state.stepsToGo() <= maxSearchDepth);

    applyAction(state, actionIndex, qValue);

}

bool DepthFirstSearch::estimateQValues(State const& state,
                                       vector<int> const& actionsToExpand,
                                       vector<double>& qValues) {
    assert(state.stepsToGo() > 0);
    assert(state.stepsToGo() <= maxSearchDepth);
    assert(qValues.size() == SearchEngine::numberOfActions);

    bool reachesGoal = false;
    for (unsigned int index = 0; index < qValues.size(); ++index) {
        if (actionsToExpand[index] == index) {
            reachesGoal |= applyAction(state, index, qValues[index]);
        }
    }
    
    return reachesGoal;
}

bool DepthFirstSearch::applyAction(State const& state, int const& actionIndex,
                                   double& reward) {
    State nxt(state.stepsToGo() - 1);
    calcStateTransition(state, actionIndex, nxt, reward);
    RandomWalk dummyEngine;

    bool reachesGoal = dummyEngine.isAGoalRewardLock(nxt);

    // Logger::logLine(nxt.toString(), Verbosity::DEBUG);
    // Logger::logLine("reward: " + to_string(reward), Verbosity::DEBUG);

    // Check if the next state is already cached
    if (DeterministicSearchEngine::stateValueCache.find(nxt) !=
        DeterministicSearchEngine::stateValueCache.end()) {
        reward += DeterministicSearchEngine::stateValueCache[nxt];
        return reachesGoal;
    }

    // Check if we have reached a leaf
    if (nxt.stepsToGo() == 1) {
        calcOptimalFinalReward(nxt, rewardHelperVar);
        reward += rewardHelperVar;
        return reachesGoal;
    }

    //  Expand the state
    double futureResult = -numeric_limits<double>::max();
    reachesGoal |= expandState(nxt, futureResult);
    reward += futureResult;

    return reachesGoal;
}

bool DepthFirstSearch::expandState(State const& state, double& result) {
    assert(!cachingEnabled ||
           (DeterministicSearchEngine::stateValueCache.find(state) ==
            DeterministicSearchEngine::stateValueCache.end()));
    assert(MathUtils::doubleIsMinusInfinity(result));

    // Get applicable actions
    vector<int> actionsToExpand = getApplicableActions(state);

    // Apply applicable actions and determine best one
    bool reachesGoal = false;
    for (unsigned int index = 0; index < actionsToExpand.size(); ++index) {
        if (actionsToExpand[index] == index) {
            double tmp = 0.0;
            reachesGoal |= applyAction(state, index, tmp);
            result = std::max(result, tmp);
        }
    }

    // Cache state value if caching is enabled
    if (cachingEnabled) {
        DeterministicSearchEngine::stateValueCache[state] = result;
    }

    return reachesGoal;
}
