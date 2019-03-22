# NUS Internship

Introduction
-
The recent interest in applying machine learning algorithms to neuroscience has led to better understanding of the human brain. Recent studies have suggests that functional connectivity from fMRI scans of the human brain provides a good way to analyse brain activity. With that in mind, this research aims to further study this relationship. 

Brain activity can be measured through Functional Connectivity(FC) data between Regions of Interests(ROIs) of the brain, when the subject is put to perform certain tasks. FC is measured in Pearson Correlation Coefficient(pcc) values, which represents the relationship between one ROI and another.  

The tasks performance of a subject can also be measured based on a specified metric. I will be using fluid intelligence as a performance indicator. 

I will be using three classical machine learning algorithms, mainly Support Vector Regression (SVR), Elastic Net (ELN) and Kernel Ridge Regression (KRR) to predict fluid intelligence using functional connectivity data.

My results demonstrated that among the 3 machine learning models used, the mean Pearson Correlation Coefficient (pcc) score of Support Vector Regression (SVR) achieves the highest accuracy, with a score of 0.272, while Elastic Net achieved a score of 0.231 and Kernel Ridge Regression achieved a score of 0.253. It is also observed that the L1 regularisation values of the training folds is significantly low, hence L2 regularisation is preferential. 

In conclusion, my results showed that there is some positive linear relationship between the predicted and the actual fluid intelligence results. This results can be further improved if there are more data provided. However, it shows that functional connectivity data has significant capability in predicting human behaviour, thus opening up the discussion on how these results can be applied to other areas in the field of neuroscience. 

Dataset
-
My dataset is obtained from the Human Connectome Project. The dataset has 954 subjects, where each subject contains an FC matrix of 419 ROIs by 419 ROIs derived from the fMRI data scans. 

My dataset however, has been prevectorised, thus a single subject will contain 87571 unique values, each representing pcc value of a ROI with respect to another. 

Fluid Intelligence
-
Fluid Intelligence is understood as the intrinsic ability of humans that are unlikely to be affected by environmental changes throughout an individual's lifetime. This can be measured by test such as pattern recognition. 

Functional Connectivity
------------------------
Functional connectivity is data from different regions of the brain, and compares the relationship between a region of the brain with the other. They usually exist as a symmetical matrix, and code needs to be written to extract the bottom half of the triangular matrix (to reduce processing time)

Methodology
------------
All 3 models are trained on Intel Core i5-5200U CPU of up to 2.7GHz, with 8GB of RAM.

The method for cross validation used in my research is also commonly known as ‘leave one out cross validation procedure’.  

In the 20 fold CV, 1 set is chosen for as the test set, and the other 19 are chosen for the training set. For the 19 sets, a 19 fold CV is done to determine the optimum hyper-parameter that leads to the highest average pearson correlation coefficient from the training set where 19 fold CV is performed. 

First, 5 hyper-parameter values were selected for tuning. Hence, each grid should contain the value of the pearson correlation coefficient from the 2 hyper-parameters inputted. The pearson correlation coefficient should not end up in the outer grids of the square (do not lie on extreme values) so that it can be known for certain that the hyper-parameters within the inner grids lead to the highest correlation. This can be done by first ensuring that the first five values of each hyper-parameter chosen covers a sufficiently large scale such that highest correlation can only lie within the inner squares and can be determining by further zooming in. 
Next, if the highest correlation is found to be in the yellow square for instance, the adjacent values are further scaled into five grids for more precise tuning. This process repeats until precise hyper-parameter values that lead to the highest correlation is obtained. 

For the purpose of this research, the precision of the hyper-parameter are set to 3 significant figures. 
Some assumptions, however, are made using this strategy. First, it is assumed that the correlation values that we obtained are global optimum and not local optimum. This could be improved by using gradient descent, but the runtime would be too long. Hence, speed is chosen over accuracy. 

Suport Vector Regression 
-

|Fold      |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 | 11 |	 12 |	13  |	14  |	15  |	16  |	17  |	18  |	19  |	20  |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|C	       | 140 | 120 | 118 | 121 | 130 | 114 | 112 | 118 | 112 | 113 |123|	109|	113|	130|	130|	146|	97|	148|	144|	101|
|Epsilon	 | 1.05| 1.10| 1.07| 0.96| 1.05| 1.06| 1.05| 1.06| 1.09| 1.12| 1.08	|1.09	|1.06	|0.99	|1.04	|0.98	|1.17	|0.87	|0.98	|0.98|
|PCC Score |0.114|0.314|-0.00366|	0.299|	0.121|	0.233|	0.297|	0.302|	0.330|	0.0756|0.397|	0.282|	0.258|	0.380|	0.382|	0.459|	0.0928|	0.426|	0.412|	0.264|
                                                Figure 3: SVR results

It can be seen from our results that the average pcc score values across the 20 folds is 0.272. 

Elastic Net
-
Fold| 	1|	2|	3|	4|	5|	6|	7|	8|	9|	10 |11|	12|	13|	14|	15|	16|	17|	18|	19|	20
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
Alpha| 	1.3|	1.9|	1.4|	1.6|	1.7|	0.8|	0.7|	0.6|	0.9|	0.7| 	0.7|	1.5|	1.2|	0.8|	0.3|	0.6|	1.6	|1.3|	1.5|	1.6
L1 ratio|	0|	0|	0|	0|	0|	0.01|	0.01|	0.01|	0.01|	0.01|	0.01|	0|	0|	0.01|	0.03|	0.01|	0|	0|	0|	0
Score| 	0.0933|	0.234|	-0.0494|	0.146|	0.0409|	0.127|	0.272|	0.321|	0.232|	0.0296| 	0.389|	0.270|	0.283|	0.331|	0.321|	0.441|	0.0864|	0.436|	0.402|	0.217
                                           Figure 4: Results of Elastic Net 

Kernel Ridge Regression
-
Fold|	1|	2|	3|	4|	5|	6|	7|	8|	9|	10| 11|	12|	13|	14|	15|	16|	17|	18|	19|	20
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
Alpha|	1132|	1606|	1043|	1353|	1428|	1369|	1261|	1268|	1769|	1331|1381|	1243|	983|	1520|	1085|	1053|	1243|	1045|	1286|	1333
Score|	0.0520|	0.213|	0.00537|	0.111|	0.414|	0.158|	0.266|	0.276|	0.237|	0.0118|0.442|	0.281|	0.237|	0.354|	0.316|	0.503|	0.158|	0.424|	0.382|	0.220
                                           Figure 5: Results of Kernel Ridge Regression

Conclusion
-
The mean scores for the Support Vector Regression, Elastic Net and Kernel Ridge Regression are 0.272, 0.231 and 0.253 respectively. This demonstrates that there is a correlation between functional connectivity data and fluid intelligence.

The results also show that among the three models used, the Support Vector Regression model is the best model to analyse the relationship between functional connectivity data and fluid intelligence- achieving the highest accuracy between predicted values and actual values of fluid intelligence data. 

It is observed that SVR model is the most computationally intensive, as more than 36 hours is required to train each fold. ELN is less computationally intensive and takes about 16 hours per fold. It is, however, least computationally intensive to train a KRR model, which takes less than 1 hour.
