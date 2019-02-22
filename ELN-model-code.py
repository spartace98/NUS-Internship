import numpy as np
import time as time

from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer

data = np.load('data_fold17.npz')
# displays the titles of the files inside the npz file
file_names = data.files
# reason for split is because train_test_subs_X[:i] is tuple not array type
training_set_X = np.concatenate(data['train_valid_x'])
# fluid intelligence at index 3 column of training set
training_set_y = np.concatenate(data['train_valid_y'])[:, 3]

print('Data Loaded')

alpha_vals = [0.6, 1.0, 1.4, 1.8, 2.2]
l1_vals = [0.00, 0.01, 0.02, 0.03, 0.04]

# creating a dataframe to store results
temp_scores = np.zeros((5, 5))
results = DataFrame(data = temp_scores, columns = C_vals, index = epsilon_vals)

def pcc_score(y, y_pred):
    return np.corrcoef(y_pred, y)[1][0]
pcc_scorer = make_scorer(pcc_score)

print('Starting Loop')

for alpha_input in alpha_vals:
    for l1_input in l1_vals:
        start = time.time()
        eln = ElasticNet(alpha = alpha_input, l1_ratio = l1_input)
        warnings.filterwarnings("ignore")
        cvs = cross_val_score(scoring = pcc_scorer, estimator = eln, X = training_set_X, y = training_set_y, cv = 19)
        mean_scores = cvs.mean()
        results[alpha_input][l1_input] = mean_scores
        end = time.time()
        duration = end - start
        print(alpha_input, l1_input, mean_scores, duration)

print(results)
