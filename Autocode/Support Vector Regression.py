import numpy as np
import time as time

from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import make_scorer

first_start = time.time()

data = np.load('data_fold1.npz')
# displays the titles of the files inside the npz file
file_names = data.files
# reason for split is because train_test_subs_X[:i] is tuple not array type
training_set_X = np.concatenate(data['train_valid_x'])
# fluid intelligence at index 3 column of training set
training_set_y = np.concatenate(data['train_valid_y'])[:, 3]
test_set_X = data['test_x']
# fluid intelligence at index 3 column of test set
test_set_Y = data['test_y'][:, 3]

# value to start loop with
C_vals = [104, 112, 120, 128, 136]
epsilon_vals = [0.90, 0.98, 1.06, 1.14, 1.22]

def pcc_score(y, y_pred):
    return np.corrcoef(y_pred, y)[1][0]
pcc_scorer = make_scorer(pcc_score)

def get_row(dataframe, val):
    for i in dataframe.columns:
        for j in dataframe.index:
            if results[i][j] == val:
                return j
    return None

def get_col(dataframe, val):
    for i in dataframe.columns:
        for j in dataframe.index:
            if results[i][j] == val:
                return i
    return None

def print_results(dataframe):
    print(dataframe)
    print(dataframe.max())
    print(dataframe.max().max())

def new_range(dataframe):
        max_val = dataframe.max().max()
        max_row, max_col = get_row(dataframe, max_val), get_col(dataframe, max_val)
        row_diff, col_diff = epsilon_vals[1] - epsilon_vals[0], C_vals[1] - C_vals[0]
        row_step, col_step = (0.5 * row_diff), (0.5 * col_diff)

        row_a, row_b = max_row - row_diff, max_row + row_diff
        col_a, col_b = max_col - col_diff, max_col + col_diff

        epsilon_vals = np.arange(row_a, row_b, row_step)
        C_vals = np.arange(col_a, col_b + 1, col_step)
        print(C_vals, epsilon_vals)
        return (C_vals, epsilon_vals)

# best case senario - find optimum value within 4 steps
for i in range(1, 5):
    # creating a dataframe to store results
    temp_scores = np.zeros((5, 5))
    results = DataFrame(data = temp_scores, columns = C_vals, index = epsilon_vals)

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

    print('Loop', i, 'is completed')
    print_results(results)

    if i < 5:
        # prints the ranges to check and generates new range to loop
        C_vals, epsilon_vals = new_range(results)

final_end = time.time()
total_duration = final_end - first_start
print('Total Time taken:', total_duration)
