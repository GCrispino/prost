#include "recommendation_function.h"

#include "thts.h"

#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"

/******************************************************************
                     Action Selection Creation
******************************************************************/

RecommendationFunction* RecommendationFunction::fromString(std::string& desc,
                                                           THTS* thts) {
    StringUtils::trim(desc);
    assert(desc[0] == '[' && desc[desc.size() - 1] == ']');
    StringUtils::removeFirstAndLastCharacter(desc);
    StringUtils::trim(desc);

    RecommendationFunction* result = nullptr;

    if (desc.find("EBA") == 0) {
        desc = desc.substr(3, desc.size());

        result = new ExpectedBestArmRecommendation(thts);
    } else if (desc.find("MPA") == 0) {
        desc = desc.substr(3, desc.size());

        result = new MostPlayedArmRecommendation(thts);
    } else {
        SystemUtils::abort("Unknown Recommendation Function: " + desc);
    }

    assert(result);
    StringUtils::trim(desc);

    while (!desc.empty()) {
        std::string param;
        std::string value;
        StringUtils::nextParamValuePair(desc, param, value);

        if (!result->setValueFromString(param, value)) {
            SystemUtils::abort("Unused parameter value pair: " + param + " / " +
                               value);
        }
    }

    return result;
}

void RecommendationFunction::printConfig(std::string indent) const {
    Logger::logLine(indent + "Recommendation function: " + name, Verbosity::VERBOSE);
}

/******************************************************************
                      Recommendation Functions
******************************************************************/

void ExpectedBestArmRecommendation::recommend(SearchNode const* rootNode,
                                              std::vector<int>& bestActions) {
    double stateValue = -std::numeric_limits<double>::max();

    std::cout << "RECOMMENDATION" << std::endl;
    std::vector<SearchNode*> const& actNodes = rootNode->children;

    for (unsigned int index = 0; index < actNodes.size(); ++index) {
        std::cout << index;
        if (actNodes[index]) {
            double reward = actNodes[index]->getExpectedRewardEstimate();
            double _reward = actNodes[index]->_getExpectedRewardEstimate();
            std::cout << ", utility estimate: " << reward << ", reward estimate: " << _reward;

            if (MathUtils::doubleIsGreater(reward, stateValue)) {
                stateValue = reward;
                bestActions.clear();
                bestActions.push_back(index);
            } else if (MathUtils::doubleIsEqual(reward, stateValue)) {
                bestActions.push_back(index);
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Best actions: " << std::endl;
    for (unsigned int index = 0; index < bestActions.size(); ++index) {
        std::cout << index << ", " << bestActions[index] << std::endl;
    }
}

void MostPlayedArmRecommendation::recommend(SearchNode const* rootNode,
                                            std::vector<int>& bestActions) {
    double stateValue = -std::numeric_limits<double>::max();

    std::vector<SearchNode*> const& actNodes = rootNode->children;

    // If one or more children are labeled as solved, MPA recommendation behaves
    // identically to EBA recommendation (this is because a solved child can not
    // be selected anymore as soon as it has been solved)
    bool solvedChildExists = false;
    for (SearchNode* child : actNodes) {
        if (child && child->solved) {
            solvedChildExists = true;
            break;
        }
    }

    for (unsigned int index = 0; index < actNodes.size(); ++index) {
        if (actNodes[index]) {
            double reward = 0.0;
            if (!solvedChildExists) {
                reward = actNodes[index]->numberOfVisits;
            } else {
                reward = actNodes[index]->getExpectedRewardEstimate();
            }

            if (MathUtils::doubleIsGreater(reward, stateValue)) {
                stateValue = reward;
                bestActions.clear();
                bestActions.push_back(index);
            } else if (MathUtils::doubleIsEqual(reward, stateValue)) {
                bestActions.push_back(index);
            }
        }
    }
}
