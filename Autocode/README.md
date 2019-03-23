DESCRIPTION
-

The scripts in this folder contain code where the results are printed automatically over time.

How it works is as follows.
1. Determine a range of values for the hyperparameters
2. If the optimal hyperparameter lies within the inner range (inner 3 boxes), the code will zoom in on these 3 values and continue
    its search for a more precise value for hyperparameter.
3. If the optimal hyperparameter, however, lies in the outer range of input hyperparameter range, the code will initialise a new range
    of hyperparameter values.
      --> there is also code to ensure that the values do not oscillate between 2 ranges (in the case where the optimal hyperparameter 
          actually lies in the outermost input range
