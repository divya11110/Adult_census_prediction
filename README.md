# Adult_census_prediction

**Objective :
To predict whether income of an individual exceeds
$50K/per year based on census data.**

**About the Dataset :**
Adult census data has 32561 entries and 15 attributes , 14
independent features and last column is the classification label
(Salary less than $50K and greater than $50k)

**Exploratory Data Analysis:**
-Counting number of Entries less than and equal to $50K is 24720
and entries with salary greater than $50K is 7841 which show that
75.9% of the entries belongs to <=$50K
While 24.8% belongs to >50K salary therefore It is an Imbalance Data

![image](https://user-images.githubusercontent.com/66716367/157060855-3434a731-34cf-4e06-a223-e18de47c6e86.png)


-There is no missing value in the dataset.
-Since education and education-number both carry similar
information hence one can be removed.
-Bar plot visualisation of numerical features are shown below.

![image](https://user-images.githubusercontent.com/66716367/157061054-20e0f347-f030-4fce-921e-b012b6ac0872.png)

The capital gain and capital loss feature histogram is right skewed
that is extreme right values at the right side of Histogram.

Box plot of Capital-gain feature

![image](https://user-images.githubusercontent.com/66716367/157061198-3ec74229-0be9-40cb-adc6-166a8d86d20a.png)

Capital gain and capital loss features have not enough information in
the quartile range hence these features can be removed.
The box plot of the Age feature shown below gives an idea that there
are outliers which need to removed

![image](https://user-images.githubusercontent.com/66716367/157061318-0fc13894-f773-4163-8de0-466809722544.png)

Age feature box plot after outliers removal

![image](https://user-images.githubusercontent.com/66716367/157061412-52b45af2-ec36-4ef3-8ab5-4678e4009b2d.png)

The Heat Map for the numerical features is shown below :

![image](https://user-images.githubusercontent.com/66716367/157061594-df693838-7234-4004-b73c-0c0a9acc700d.png)

Categorical features like native-country, work class, education ,
marital status, occupation, relationship, race , sex, native country is
converted to numerical value using Label Encoder.
Heat Map shown below after encoding the categorical features :

![image](https://user-images.githubusercontent.com/66716367/157061668-a5e444aa-137e-49bb-b95f-30c1d0362b8c.png)

Since correlation is very less it shows that features are not dependent
on each other.

We have encoded our dependent variable salary as 1 for salary
<=$50K and 0 for salary >$50K
After Data preprocessing now we can use the independent features
to train our machine learning model.

Before training the model we need to transform the features by
scaling each feature to a given range. I have used MinMaxScaler for
feature scaling. It scales all the features in the range [0,1] or [-1,1] if
negative values in the data.

Since data is not normally distributed I have not opted for Standard
Scaler.

I have used SVM(poly kernel) and Xgboost models.

Xgboost classification report on test data

![image](https://user-images.githubusercontent.com/66716367/157061764-78164e14-f349-4792-859e-c1e4f9e2d82e.png)

Since the data is imbalanced we should go for F1 score weighted
metrics to evaluate the model performance.

SVM poly kernel gave 79.9% f1 score on testing data . Xgboost 81% f1
score. Xgboost gave better results because it automatically takes
care of missing values while SVM need to first impute the missing
value before putting the data into it. Xgboost has inbuilt
regularisation thus prevents overfitting that is the reason why
Xgboost performs best on test data as compared to svm
