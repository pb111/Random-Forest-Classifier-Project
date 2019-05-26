# Random Forest Classifier with Python and Scikit-Learn


Random Forest is a supervised machine learning algorithm which is based on ensemble learning. In this project, I build two Random Forest Classifier models to predict the safety of the car, one with 10 decision-trees and another one with 100 decision-trees. The expected accuracy increases with number of decision-trees in the model. I have demonstrated the **feature selection process** using the Random Forest model to find only the important features, rebuild the model using these features and see its effect on accuracy. I have used the **Car Evaluation Data Set** for this project, downloaded from the UCI Machine Learning Repository website.


===============================================================================


## Table of Contents


1.	Introduction to Random Forest algorithm

2.	Random Forest classification intuition

3.	Advantages and disadvantages of Random Forest algorithm

4.	Feature selection with Random Forest algorithm

5.	Difference between Random Forests and Decision Trees

6.	Relationship to nearest neighbours

7.	The problem statement

8.	Results and conclusion

9.	Applications of Random Forest classification 

10.	References


===============================================================================


## 1. Introduction to Random Forest algorithm

Random forest is a supervised learning algorithm. It has two variations – one is used for classification problems and other is used for regression problems. It is one of the most flexible and easy to use algorithm. It creates decision trees on the given data samples, gets prediction from each tree and selects the best solution by means of voting. It is also a pretty good indicator of feature importance.
Random forest algorithm combines multiple decision-trees, resulting in a forest of trees, hence the name “Random Forest”. In the random forest classifier, the higher the number of trees in the forest results in higher accuracy


===============================================================================


## 2. Random Forest algorithm intuition


Random forest algorithm intuition can be divided into two stages. 
In the first stage, we randomly select “k” features out of total “m” features and build the random forest. In the first stage, 
we proceed as follows:-


1.	Randomly select “k” features from a total of “m” features where k < m.

2.	Among the “k” features, calculate the node “d” using the best split point.

3.	Split the node into daughter nodes using the best split.

4.	Repeat 1 to 3 steps until “l” number of nodes has been reached.

5.	Build forest by repeating steps 1 to 4 for “n” number of times to create “n” number of trees.


In the second stage, we make predictions using the trained random forest algorithm. 


1.	We take the test features and use the rules of each randomly created decision tree to predict the outcome and stores the predicted outcome.

2.	Then, we calculate the votes for each predicted target.

3.	Finally, we consider the high voted predicted target as the final prediction from the random forest algorithm.


===============================================================================


## 3. Advantages and disadvantages of Random Forest algorithm


The advantages of Random forest algorithm are as follows:-


1.	Random forest algorithm can be used to solve both classification and regression problems.

2.	It is considered as very accurate and robust model because it uses large number of decision-trees to make predictions.

3.	Random forests takes the average of all the predictions made by the decision-trees, which cancels out the biases. So, it does not suffer from the overfitting problem. 

4.	Random forest classifier can handle the missing values. There are two ways to handle the missing values. First is to use median values to replace continuous variables and second is to compute the proximity-weighted average of missing values.

5.	Random forest classifier can be used for feature selection. It means selecting the most important features out of the available features from the training dataset.


The disadvantages of Random Forest algorithm are listed below:-


1.	The biggest disadvantage of random forests is its computational complexity. Random forests is very slow in making predictions because large number of decision-trees are used to make predictions. All the trees in the forest have to make a prediction for the same input and then perform voting on it. So, it is a time-consuming process.

2.	The model is difficult to interpret as compared to a decision-tree, where we can easily make a prediction as compared to a decision-tree.


===============================================================================


## 4. Feature selection with Random Forest algorithm


Random forests algorithm can be used for feature selection process. This algorithm can be used to rank the importance of variables in a regression or classification problem. We measure the variable importance in a dataset by fitting the random forest algorithm to the data. During the fitting process, the out-of-bag error for each data point is recorded and averaged over the forest. 


The importance of the j-th feature was measured after training. The values of the j-th feature were permuted among the training data and the out-of-bag error was again computed on this perturbed dataset. The importance score for the j-th feature is computed by averaging the difference in out-of-bag error before and after the permutation over all trees. The score is normalized by the standard deviation of these differences.


Features which produce large values for this score are ranked as more important than features which produce small values. Based on this score, we will choose the most important features and drop the least important ones for model building. 


===============================================================================


## 5. Difference between Random Forests and Decision Trees


I will compare random forests with decision-trees. Some salient features of comparison are as follows:-


1.	Random forests is a set of multiple decision-trees.

2.	Decision-trees are computationally faster as compared to random forests.

3.	Deep decision-trees may suffer from overfitting. Random forest prevents overfitting by creating trees on random forests.

4.	Random forest is difficult to interpret. But, a decision-tree is easily interpretable and can be converted to rules.


===============================================================================


## 6. Relationship to nearest neighbours


A relationship between random forests and the k-nearest neighbours algorithm was pointed out by Lin and Jeon in 2002. It turns out that both can be viewed as so-called `weighted neighbourhoods schemes`. These are models built from a training set that make predictions for new points by looking at the neighbourhood of the point, formalized by a weight function.


===============================================================================


## 7. The problem statement


The problem is to predict the safety of the car. In this project, I build a Random Fores t Classifier to predict the safety of the car. I implement Random Forest Classification with Python and Scikit-Learn. I have used the **Car Evaluation Data Set** for this project, downloaded from the UCI Machine Learning Repository website.


The dataset can be found at the following url-


http://archive.ics.uci.edu/ml/datasets/Car+Evaluation


===============================================================================


## 8. Results and conclusion



===============================================================================



## 9. Applications of Random Forest classification


Random forests has a wide variety of applications. Its applications include -


1.	Recommendation engines

2.	Image classification

3.	Feature selection

4.	Identify frauds

5.	Predict diseases

6.	Stock-market

7.	E-commerce


===============================================================================


## 10. References


The work done in this project is inspired from following books and websites:-

1.	Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron

2.	Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido

3.	Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves

4.	https://en.wikipedia.org/wiki/Random_forest

5.	https://www.datacamp.com/community/tutorials/random-forests-classifier-python

6.	http://dataaspirant.com/2017/05/22/random-forest-algorithm-machine-learing/

7.	https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/









