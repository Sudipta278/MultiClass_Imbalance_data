#!/usr/bin/env python
# coding: utf-8

# **Summary of the dataset and Goal of the Project**

# This datasets is related to variants of the Portuguese "Vinho Verde" wine.The dataset describes the amount of various chemicals present in wine and their effect on it's quality. 
# 
# The Goal is to anticipate the quality of wine.

# In[1]:


# import all libraries and dependencies for dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',None)
dfr = pd.read_csv("winequality-red.csv",sep=";")
dfr.head()


# In[2]:


pd.set_option('display.max_columns',None)
dfw = pd.read_csv("winequality-white.csv",sep=";")
dfw.head()


# In[3]:


#information of the data
dfr.info()
dfw.info()


# In[4]:


# Total count of 'Outcome'
print(dfw['quality'].value_counts())
print(dfr['quality'].value_counts())


# In[5]:


dfw['Type']='white'
dfr['Type']='red'


# In[6]:


df=pd.concat([dfw,dfr])
df.info()


# In[7]:


df.head(7)


# **Data Preprocessing**

# We have investigated below
# 1. No null value
# 2. Only 1 categorical column

# In[8]:


# Total count of 'quality'
print(df['quality'].value_counts())
sns.countplot(x="quality",data=df)


# Findings: This dataset has imbalance class of wine quality. Later we will do SMOTE to handle the imbalance classes.

# In[9]:


df.describe()


# **Exploratory Data Analysis (EDA)**

# In[10]:


# The Heatmap will show us the relationship between two variables
import numpy as np
plt.figure(figsize=(15, 8))
mask=np.triu(np.ones_like(df.corr()))
sns.heatmap(df.corr(),cmap='coolwarm',mask=mask,annot=True)
plt.title('Correlation Matrix')
plt.show()


# Findings: 
# 1. The heatmap indicates a negative correlation between wine quality and fixed acidity, volatile acidity, residual sugar, chlorides, total sulfur dioxide, and density. However, the Pearson correlation coefficient values suggest that the relationship is not very strong.
# 2. There are some positive correlation between wine quality and citric acid, free sulfur dioxide, pH, sulphates, alcohol. However r value suggest a strong relationship with alcohol only.
# 3. In the heat map also suggest a strong correlation between total sulfur dioxide and free sulfur dioxide. 

# In[11]:


# Checking correlation between total sulfur dioxide and free sulfur dioxide
from scipy import stats
sns.scatterplot(data=df,x="free sulfur dioxide",y="total sulfur dioxide")
slope, intercept, r_value, p_value, std_err = stats.linregress(df["free sulfur dioxide"], df["total sulfur dioxide"])
plt.plot(df["free sulfur dioxide"], slope * df["free sulfur dioxide"] + intercept, color='blue')
plt.show()


# In[12]:


# Checking impact of quantity of alcohol in the quality of wines
sns.barplot(x="quality",y="alcohol",data=df)
df.groupby("quality")['alcohol'].mean()


# Findings: 
# We have quality wine from 3 to 9 irrespective of types of wine in this dataset. Here the avg. quantity of alcohol is in a range of 9.83 to 12.18 where wine quality grade 5 contain the min alcohol and the quality grade 9 contain the max alcohol.

# In[13]:


# Checking alcohol data distribution in terms of quality wine
sns.boxplot(data=df,x="quality",y="alcohol")


# Findings: 
# The above boxplot shows outliers in wine quality grade 5 which may impact on the avg alcohol contain compare to other grades.

# In[14]:


sns.barplot(x="Type",y="quality",data=df)
df.groupby("Type")['quality'].mean()


# Finding: Avg quality of both white & red wines are in between 5.6 to 5.8. According to the data, quality of both type of wines are same.

# In[15]:


# Hypothesis Testing
#H0:There is no difference in quality of white & red wines.
#Ha:There is a significance significant difference in quality of white & red wines.
# Significant level is 0.05
# Check p-value
from scipy import stats
white=df[df['Type']=="white"]
red=df[df['Type']=="red"]
sample_white=white.sample(n=20,random_state=200,replace=True)
sample_red=red.sample(n=20,random_state=100,replace=True)
t_statistic, p_value = stats.ttest_ind(a=sample_white['quality'], b=sample_red['quality'],equal_var=False)
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# Finding: Here p-value is greater than significant level.Hence we fail to reject the null hypothesis. 
# There is no difference in quality of white & red wines based on alcohol.

# In[16]:


# Use SMOTE for the imbalance class of data 
from imblearn.over_sampling import SMOTE
X = df.drop(['Type','quality'], axis=1)
y = df['quality']
sm = SMOTE(random_state=42, k_neighbors=4)
X_res, y_res = sm.fit_resample(X, y)


# In[17]:


# Checking final result
idx,c=np.unique(y_res,return_counts=True)
sns.barplot(x=idx,y=c)


# 1. Here the outcome dependent variable is categorical. Hence we may consider logistic regression model.
# 2. We found a strong positive correlation between total sulfur dioxide and free sulfur dioxide hence to follow 'no multicollinearity' we will drop one variable. Free sulfur dioxide has higher value of r compare to total sulfur dioxide with outcome variable. Hence we will drop total sulfur dioxide.

# In[18]:


X = df.drop(['Type','quality','total sulfur dioxide'], axis=1)
y = df['quality']


# **Data Model**

# In[19]:


# LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Build regression model
LR=LogisticRegression()
LR.fit(X_train,y_train)

# Save the prediction
y_LR_pred=LR.predict(X_test)
print(classification_report(y_test, y_LR_pred))


# In[20]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)

# Save the prediction
y_gnb_pred=gnb.predict(X_test)
print(classification_report(y_test, y_gnb_pred))


# In[31]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

# Save the prediction
y_dt_pred=dt.predict(X_test)
print(classification_report(y_test, y_dt_pred))


# In[32]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)

# Save the prediction
y_rf_pred=dt.predict(X_test)
print(classification_report(y_test, y_rf_pred))


# In[33]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_rf_pred)
disp=metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# **Model Evaluation & Hyper Parameter Tuning**

# In[34]:


from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

dtc = DecisionTreeClassifier()

cv_params=[{'max_depth':[2,4,5,6,7,8,9,10],'min_samples_leaf':[2,5,10,20,30,40,50]}]

clf = GridSearchCV(dtc, cv_params, cv = 10, scoring='accuracy')

clf.fit(X_train,y_train)

print(clf.best_params_)

print(clf.best_score_)


# In[35]:


# Create a new Decision Tree classifier with the best parameters
best_dtc = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2)

# Fit the new classifier on the training data
best_dtc.fit(X_train, y_train)

# Make predictions on the test data
y_best_dtc_pred = best_dtc.predict(X_test)

# Evaluate the performance of the new classifier
print(classification_report(y_test, y_best_dtc_pred))


# Conclusion: Both models perform reasonably well with an accuracy of 81%. The precision, recall, and F1-score are consistent across different classes for both models. Given the similarity in performance metrics, it's difficult to determine which model is definitively better based solely on this information. However, in practice, Random Forest models tend to be more robust and less prone to overfitting compared to single Decision Trees, especially for complex datasets. Therefore, if computational resources allow, the Random Forest model might be preferred for its potential to generalize better to unseen data and handle more complex relationships in the data.
