#ifndef BACKUP_FUNCTION_H
#define BACKUP_FUNCTION_H

#include <string>
#include "utils/math_utils.h"

class THTS;
class SearchNode;

/******************************************************************
                         Backup Function
******************************************************************/

class BackupFunction {
public:
    virtual ~BackupFunction() {}

    // Create a backup function component
    static BackupFunction* fromString(std::string& desc, THTS* thts);

    // Set parameters from command line
    virtual bool setValueFromString(std::string& /*param*/,
                                    std::string& /*value*/) {
        return false;
    }

    // This is called when caching is disabled because memory becomes sparse
    virtual void disableCaching() {}

    virtual void initSession() {}
    virtual void initRound() {}
    virtual void finishRound() {}
    virtual void initStep() {
        // Reset per step statistics
        skippedBackups = 0;
    }
    virtual void finishStep() {}
    virtual void initTrial() {
        lockBackup = false;
    }

    // Backup functions
    virtual void backupDecisionNodeLeaf(SearchNode* node,
                                        double const& futReward,
                                        float k_g = 1,
                                        bool reachedGoal = false);
    virtual void backupDecisionNode(SearchNode* node, bool reachedGoal = false);
    virtual void backupChanceNode(SearchNode* node,
                                  double const& futReward,
                                        float k_g = 1,
                                        bool reachedGoal = false) = 0;

    // Prints statistics
    virtual void printConfig(std::string indent) const;
    virtual void printStepStatistics(std::string indent) const;
    virtual void printRoundStatistics(std::string /*indent*/) const {}
    
protected:
    BackupFunction(THTS* _thts,
                   std::string _name,
                   bool _useSolveLabeling = false,
                   bool _useBackupLock = false,
                   float (*_utility_function) (float) = MathUtils::u)
        : thts(_thts),
          name(_name),
          useSolveLabeling(_useSolveLabeling),
          useBackupLock(_useBackupLock),
          utility_function(_utility_function) {}

    THTS* thts;

    // Name, used for output only
    std::string name;

    // If this is true, no further nodes a rebacked up in this trial
    bool lockBackup;

    // Parameter
    bool useSolveLabeling;
    bool useBackupLock;

    // Per step statistics
    int skippedBackups;

    // Tests which access private members
    friend class BFSTestSearch;

    // Utility function
    float (*utility_function)(float);
};

/******************************************************************
                       Monte-Carlo Backups
******************************************************************/

class MCBackupFunction : public BackupFunction {
public:
    MCBackupFunction(THTS* _thts)
        : BackupFunction(_thts, "MonteCarlo backup"),
          initialLearningRate(1.0),
          learningRateDecay(1.0) {}

    // Set parameters from command line
    bool setValueFromString(std::string& param, std::string& value) override;

    // Parameter setter
    void setInitialLearningRate(double _initialLearningRate) {
        initialLearningRate = _initialLearningRate;
    }

    void setLearningRateDecay(double _learningRateDecay) {
        learningRateDecay = _learningRateDecay;
    }

    // Backup functions
    void backupChanceNode(SearchNode* node, double const& futReward,
                                        float k_g = 1,
                                        bool reachedGoal = false) override;

    // Prints statistics
    void printConfig(std::string indent) const override;

private:
    double initialLearningRate;
    double learningRateDecay;
};

/******************************************************************
                   MaxMonte-Carlo Backups
******************************************************************/

class MaxMCBackupFunction : public BackupFunction {
public:
    MaxMCBackupFunction(THTS* _thts) : BackupFunction(_thts, "MaxMonteCarlo backup") {}

    // Backup functions
    void backupChanceNode(SearchNode* node, double const& futReward,
                                        float k_g = 1,
                                        bool reachedGoal = false) override;
};

/******************************************************************
                      Partial Bellman Backups
******************************************************************/

class PBBackupFunction : public BackupFunction {
public:
    PBBackupFunction(THTS* _thts) : BackupFunction(_thts, "PartialBellman backup", true, true) {}

    // Backup functions
    void backupChanceNode(SearchNode* node, double const& futReward,
                                        float k_g = 1,
                                        bool reachedGoal = false) override;
};

#endif
