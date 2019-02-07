# NUS Internship

Introduction
------------
The recent interest in applying machine learning algorithms to neuroscience has led to better understanding of the human brain. Recent studies have suggests that functional connectivity from fMRI scans of the human brain provides a good way to analyse brain activity. With that in mind, this research aims to further study this relationship. We will be using fluid intelligence as a behavioural measure of brain activity. We will also be using three machine learning algorithms, mainly Support Vector Regression (SVR), Elastic Net (ELN) and Kernel Ridge Regression (KRR) to predict fluid intelligence using functional connectivity data. 

This research aims to compare 3 common machine learning models and their effectiveness in predicting fluid intelligence of individuals from functional connectivity data in the HCP dataset. However, to determine the best model, the optimal hyperparameter of each model has to be determined, and then applied to the model to determine determine its effectiveness in prediciting never seen before test set results. 

Fluid Intelligence
------------------
Fluid Intelligence is understood as the intrinsic ability of humans that are unlikely to be affected by environmental changes throughout an individual's lifetime. This can be measured by test such as pattern recognition. 

Functional Connectivity
------------------------
Functional connectivity is data from different regions of the brain, and compares the relationship between a region of the brain with the other. They usually exist as a symmetical matrix, and code needs to be written to extract the bottom half of the triangular matrix (to reduce processing time)

Methodology
------------
In the 20 fold CV, 1 set is chosen for as the test set, and the other 19 are chosen for the training set. For the 19 sets, a 19 fold CV is done to determine the optimum hyper-parameter that leads to the highest average pearson correlation coefficient from the training set where 19 fold CV is performed. 

First, 5 hyper-parameter values were selected for tuning. Hence, each grid should contain the value of the pearson correlation coefficient from the 2 hyper-parameters inputted. The pearson correlation coefficient should not end up in the outer grids of the square (do not lie on extreme values) so that it can be known for certain that the hyper-parameters within the inner grids lead to the highest correlation. This can be done by first ensuring that the first five values of each hyper-parameter chosen covers a sufficiently large scale such that highest correlation can only lie within the inner squares and can be determining by further zooming in. 
Next, if the highest correlation is found to be in the yellow square for instance, the adjacent values are further scaled into five grids for more precise tuning. This process repeats until precise hyper-parameter values that lead to the highest correlation is obtained. 
For the purpose of this research, the precision of the hyper-parameter are set to 3 significant figures. 
Some assumptions, however, are made using this strategy. First, it is assumed that the correlation values that we obtained are global optimum and not local optimum. This could be improved by using gradient descent, but the runtime would be too long. Hence, speed is chosen over accuracy. 

Support Vector Regression
----------------------
Support Vector Regression is a form of supervised machine learning model which predicts the values of data based on pre-determined results. The Support Vector Machine maps the multidimensional data points in space. Using the kernel trick (linear, rbf, poly etc), a plane in the high dimensional space that best improves the scoring metric is determined and then casted in 2 dimensional axes.

The hyperparameters in SVR are 
1. Kernel
2. C
3. Epsilon

In the instance of this research, the rbf kernel has been found to lead to more precise predictions. Hence, the rbf kernel is chosen, and the hyperparamters used for tuning are C and epsilon. 


Elastic Net CV
----------------------


Kernel Ridge Regression
----------------------


