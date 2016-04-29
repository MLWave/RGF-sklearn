# Regularized Greedy Forest Wrappers

First version for a toy Scikit/learn API compatible wrapper for Regularized Greedy Forests [Johnson & Zhang, 2014]

## Usage

### Classification

**RegularizedGreedyForestClassifier(verbose=0, max_leaf=500, test_interval=100, loc_exec=loc_exec, loc_temp=loc_temp, algorithm="RGF", loss="LS", l2="1", prefix="model")**

Parameter|Description
---|---
verbose|Int. Verbosity of the classifier. *Default=0*
max_leaf|Int. Max number of leafs to create before halting. *Default=500*
test_interval|Int. Save models during intervals. *Default=100*
algorithm|String. Any of `RGF` (RGF with L2 regularization, `RGF_Opt` (RGF with min-penalty regularization), `RGF_Sib` (RGF with min-penalty regularization with sum-to-zero sibling constraints) *Default=RGF*
loss|String. Any of `LS` (Least squares), `Expo` (Exponential), `Log` (Logarithmic). *Default=LS*
L2|Float. Amount of L2 regularization. `1.0`, `0.1` and `0.01` are sane values. *Default=1.0*

### Regression

**RegularizedGreedyForestRegressor(verbose=0, max_leaf=500, test_interval=100, loc_exec=loc_exec, loc_temp=loc_temp, algorithm="RGF", loss="LS", l2="1", prefix="model")**

Parameter|Description
---|---
verbose|Int. Verbosity of the regressor. *Default=0*
max_leaf|Int. Max number of leafs to create before halting. *Default=500*
test_interval|Int. Save models during intervals. *Default=100*
algorithm|String. Any of `RGF` (RGF with L2 regularization, `RGF_Opt` (RGF with min-penalty regularization), `RGF_Sib` (RGF with min-penalty regularization with sum-to-zero sibling constraints) *Default=RGF*
loss|String. Any of `LS` (Least squares), `Expo` (Exponential), `Log` (Logarithmic). *Default=LS*
L2|Float. Amount of L2 regularization. `1.0`, `0.1` and `0.01` are sane values. *Default=1.0*