# ELASTIC NET CV AUTOCODE
import numpy as np
import time as time
import warnings

from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer

data = np.load('data_fold5.npz')
# reason for split is because train_test_subs_X[:i] is tuple not array type
training_set_X = np.concatenate(data['train_valid_x'])
# fluid intelligence at index 3 column of training set
training_set_y = np.concatenate(data['train_valid_y'])[:, 3]

print('Data Loaded')

alpha_vals = [0.6, 1.0, 1.4, 1.8, 2.2]
l1_vals = [0.00, 0.01, 0.02, 0.03, 0.04]

# creating a dataframe to store results
temp_scores = np.zeros((5,5))
results = DataFrame(data = temp_scores, columns = alpha_vals, index = l1_vals)

def pcc_score(y, y_pred):
    return np.corrcoef(y_pred, y)[1][0]

pcc_scorer = make_scorer(pcc_score)

def get_hyperparameters(dataframe, val):
    for i in dataframe.columns:
        for j in dataframe.index:
            if dataframe[i][j] == val:
                return i, j
    return None

def indiv_score(alpha_vals, l1_vals, results):
    for alpha_input in alpha_vals:
        for l1_input in l1_vals:
            start = time.time()
            eln = ElasticNet(alpha = alpha_input, l1_ratio = l1_input)
            warnings.filterwarnings("ignore", category = DeprecationWarning)
            cvs = cross_val_score(scoring = pcc_scorer, estimator = eln, X = training_set_X, y = training_set_y, cv = 19)
            mean_scores = cvs.mean()
            results[alpha_input][l1_input] = mean_scores
            end = time.time()
            duration = end - start
            print(alpha_input, l1_input, mean_scores, duration)
    
    return results

# based on the best alpha value and the previous alpha inputs, evaluates the next set of alpha inputs
# l1 ratio will always remain the same
def new_array(temp_alpha, alpha_vals):
    old_alpha_interval = alpha_vals[1] - alpha_vals[0]
    alpha_range_min, alpha_range_max = temp_alpha - old_alpha_interval, temp_alpha + old_alpha_interval
    new_alpha_interval = old_alpha_interval / 2
    new_alpha_vals = np.arange(alpha_range_min, (alpha_range_max + 0.1), new_alpha_interval)

    # assume results are in centre of results dataframe
    # not applicable because result will either be 0 or near zero
    # alpha_range_min, alpha_range_max = temp_alpha - old_alpha_interval, temp_alpha + old_alpha_interval    
    # l1_range_min, l1_range_max = temp_l1 - old_l1_interval, temp_l1 + old_l1_interval    
    # new_alpha_interval = new_alpha_interval / 2
    # new_l1_interval = new_l1_interval / 2
    # new_alpha_vals = np.arange(alpha_range_min, (alpha_range_max + 0.1), new_alpha_interval)
    # new_l1_vals = np.arange(l1_range_min, (l1_range_max + 0.001), new_l1_interval)

    # reassign the variables for results with hyperparameters are extreme ends
    # if the best alpha values were at the extreme ends of the alpha inputs

    if temp_alpha == alpha_vals[0]:
        new_alpha_interval = old_alpha_interval
        new_alpha_vals = np.arange(alpha_vals[1] - new_alpha_interval * 4, alpha_vals[1], new_alpha_interval)


    elif temp_alpha == alpha_vals[-1]:
        new_alpha_interval = old_alpha_interval
        new_alpha_vals = np.arange(alpha_vals[-2], alpha_vals[-2] + new_alpha_interval * 4, new_alpha_interval)

    # if the best l1 values were at the extreme ends of the l1 inputs

    return (new_alpha_interval, new_alpha_vals)

# evaluates and prints out the result dataframe, best score and best alpha value for each loop
# returns the best alpha value 
def get_results(alpha_vals, l1_vals, results):
    temp_results = indiv_score(alpha_vals, l1_vals, results)
    best_score = temp_results.max().max()
    temp_alpha, temp_l1 = get_hyperparameters(temp_results, best_score)

    print(temp_results)
    print('Best Pearson Score:', best_score)
    print('Best Alpha Value:', temp_alpha)
    print('Best l1 Value:', temp_l1)        

    return temp_alpha

# if the difference betweeen 2 consec values are more than 1, continue the loop
	# refreshes the dataframe
	# evaluates the scores for each alpha
	# stores in the dataframe
	# finds the best score
	# create new dataframe and start over
def best_results(alpha_vals, l1_vals):
    temp_scores = np.zeros((5,5))
    results = DataFrame(data = temp_scores, columns = alpha_vals, index = l1_vals)
    
    if alpha_vals[1] - alpha_vals[0] > 0.1:
        temp_alpha = get_results(alpha_vals, l1_vals, results)
    
        new_alpha_interval, new_alpha_vals = new_array(temp_alpha, alpha_vals)
        print('New Alpha Interval:', new_alpha_interval)
        print('New Alpha Values:', new_alpha_vals)

        # begins new loop with new alpha vals
        best_results(new_alpha_vals, l1_vals)

    else:
        get_results(alpha_vals, l1_vals, results)

# Officially Starts the Loop
print('Starting analysis')
best_results(alpha_vals, l1_vals) 