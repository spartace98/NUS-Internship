import numpy as np
import time as time

from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import make_scorer

data = np.load('data_fold17.npz')
# displays the titles of the files inside the npz file
file_names = data.files
# reason for split is because train_test_subs_X[:i] is tuple not array type
training_set_X = np.concatenate(data['train_valid_x'])
# fluid intelligence at index 3 column of training set
training_set_y = np.concatenate(data['train_valid_y'])[:, 3]

print('Data Loaded')

C_vals = [92, 96, 100, 104, 108]
epsilon_vals = [1.08, 1.12, 1.16, 1.20, 1.24]

# creating a dataframe to store results
temp_scores = np.zeros((5, 5))
results = DataFrame(data = temp_scores, columns = C_vals, index = epsilon_vals)

def pcc_score(y, y_pred):
    return np.corrcoef(y_pred, y)[1][0]
pcc_scorer = make_scorer(pcc_score)

print('Starting Loop')

for C_input in C_vals:
    for epsilon_input in epsilon_vals:
    	# list will refresh for every new epsilon (and C_input) value
        start = time.time()
        svr = SVR(C = C_input, epsilon = epsilon_input)
        cvs = cross_val_score(scoring = pcc_scorer, estimator = svr, X = training_set_X, y = training_set_y, cv = 19)
        results[C_input][epsilon_input] = cvs.mean()
        end = time.time()
        duration = end - start
        print(C_input, epsilon_input, cvs.mean(), duration)

print(results)
