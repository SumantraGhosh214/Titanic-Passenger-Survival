#!/usr/bin/env python
# coding: utf-8

# The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on.

# In[1]:


#First, let's import the dataset from our lab.
import os
TITANIC_PATH='/cxldata/datasets/project/titanic'
TITANIC_PATH


# In[2]:


#Now we divide the dataset into train and test set.
import pandas as pd
def load_titanic_data(filename,titanic_path=TITANIC_PATH):
    csv_path=os.path.join(titanic_path,filename)
    return pd.read_csv(csv_path)

train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")


# In[6]:


#Now we look into the training data and analyze it.
train_data.head()


# In[7]:


train_data.info() 


# From above output we now know 2 columns (Age,Cabin) have missing values

# In[8]:


#Let's take a look at the numerical attributes of the training data:
train_data.describe()


# This o/p tells us info such as mean age of passangers(29.699118) , maximum fare (512.329200)

# In[9]:


#Finding number of female passengers on-board from the training dataset.
train_data["Sex"].value_counts()[1]


# In[10]:


#Number of male passengers on-board from the training dataset.
train_data["Sex"].value_counts()[0]


# In[11]:


#Now let's build our preprocessing pipelines.
from sklearn.base import BaseEstimator,TransformerMixin


# In[12]:


#Code for creating pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[13]:


#Let's build the pipeline for the numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])


# In[14]:


#fitting the training data in this numerical pipeline:
num_pipeline.fit_transform(train_data)


# In[15]:


#We will also need an imputer for the string categorical columns (the regular SimpleImputer does not work on those):
class MostFrequentImputer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[16]:


#Now let us build the pipeline for the categorical attributes:
from sklearn.preprocessing import OneHotEncoder
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])


# In[17]:


#Now fitting the training data into catagorical pipeline:

cat_pipeline.fit_transform(train_data)


# In[18]:


#Now let's join the numerical and categorical pipelines.
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[19]:


#Fit the training data in this pipeline:
X_train = preprocess_pipeline.fit_transform(train_data)

#Save the labels in y_train:
y_train = train_data["Survived"]


# In[20]:


#Now train an SVC classifier on the training set.
from sklearn.svm import SVC
svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train, y_train)


# In[21]:


#Use the model created in the previous step to make 
#predictions on the test set:

X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)


# In[22]:


# Now we will evaluate our SVC model.
from sklearn.model_selection import cross_val_score
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()


# In[23]:


#The score we got with the SVC classifier was not that good. 
#So now we will train and predict the dataset using 
#a RandomForest classifier.

from sklearn.ensemble import RandomForestClassifier

#Train and evaluate the RandomForestClassfier:
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()


# In[ ]:




