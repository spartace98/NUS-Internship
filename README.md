# NUS Internship

Introduction
------------
The recent interest in applying machine learning algorithms to neuroscience has led to better understanding of the human brain. Recent studies have suggests that functional connectivity from fMRI scans of the human brain provides a good way to analyse brain activity. With that in mind, this research aims to further study this relationship. 

Brain activity can be measured through Functional Connectivity(FC) data between Regions of Interests(ROIs) of the brain, when the subject is put to perform certain tasks. FC is measured in Pearson Correlation Coefficient(pcc) values, which represents the relationship between one ROI and another.  

The tasks performance of a subject can also be measured based on a specified metric. I will be using fluid intelligence as a performance indicator. 

I will be using three classical machine learning algorithms, mainly Support Vector Regression (SVR), Elastic Net (ELN) and Kernel Ridge Regression (KRR) to predict fluid intelligence using functional connectivity data.

My results demonstrated that among the 3 machine learning models used, the mean Pearson Correlation Coefficient (pcc) score of Support Vector Regression (SVR) achieves the highest accuracy, with a score of 0.272, while Elastic Net achieved a score of 0.231 and Kernel Ridge Regression achieved a score of 0.253. It is also observed that the L1 regularisation values of the training folds is significantly low, hence L2 regularisation is preferential. 

In conclusion, my results showed that there is some positive linear relationship between the predicted and the actual fluid intelligence results. This results can be further improved if there are more data provided. However, it shows that functional connectivity data has significant capability in predicting human behaviour, thus opening up the discussion on how these results can be applied to other areas in the field of neuroscience. 

Dataset
---------
My dataset is obtained from the Human Connectome Project. The dataset has 954 subjects, where each subject contains an FC matrix of 419 ROIs by 419 ROIs derived from the fMRI data scans. 

My dataset however, has been prevectorised, thus a single subject will contain 87571 unique values, each representing pcc value of a ROI with respect to another. 

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

