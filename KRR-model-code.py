# KERNEL RIDGE REGRESSION PROGRAM

import numpy as np
import time as time

from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer

data = np.load('data_fold1.npz')
# displays the titles of the files inside the npz file
file_names = data.files
# reason for split is because train_test_subs_X[:i] is tuple not array type
training_set_X = np.concatenate(data['train_valid_x'])
# fluid intelligence at index 3 column of training set
training_set_y = np.concatenate(data['train_valid_y'])[:, 3]

print('Data Loaded')

alpha_vals = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]

# creating a dataframe to store results
temp_scores = np.zeros((1,10))
results = DataFrame(data = temp_scores, columns = alpha_vals, index = ['score'])

def pcc_score(y, y_pred):
    return np.corrcoef(y_pred, y)[1][0]
pcc_scorer = make_scorer(pcc_score)

print('Starting Loop')

for alpha_input in alpha_vals:
	# list will refresh for every new alpha_input value
    start = time.time()
    krr = KernelRidge(alpha = alpha_input)
    cvs = cross_val_score(scoring = pcc_scorer, estimator = krr, X = training_set_X, y = training_set_y, cv = 19)
    results[alpha_input] = cvs.mean()
    end = time.time()
    duration = end - start
    print(alpha_input, results[alpha_input], duration)

print(results)
