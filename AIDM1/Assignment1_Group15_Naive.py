#!/usr/bin/env python
# coding: utf-8

# ## **Advances in Data Mining**
# 
# Stephan van der Putten | (s1528459) | stvdputtenjur@gmail.com  
# Theo Baart | s2370328 | s2370328@student.leidenuniv.nl
# 
# ### **Assignment 1**
# This assignment is concered with implementing formulas and models capable of predicting movie ratings for a set of users. Additionally, the accuracy of the various models are checked.
# 
# Note all implementations are based on the assignment guidelines and helper files given as well as the documentation of the used functions.
# 
# #### **Naive Approaches**
# This specific notebook handles the implementation of various naive approaches/formulas to the prediction problem.

# The following snippet handles all imports.

# In[618]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import sklearn
from sklearn.model_selection import KFold


# ### **Data Extraction and Preparation**
# 
# The `convert_data` function is used to extract the data from the raw data file and store it in a format that is more convenient for us. 
# 
# In order to do this the function uses the following parameters:
#   * `path` - the (relative) location of the raw dataset (with filetype `.dat`)
#   * `cols` - which columns to load from the raw dataset
#   * `delim` - the delimitor used in the raw dataset
#   * `dt` - the datatype used for the data in the raw dataset
#     
# Additionally it returns the following value:
#   * `path` - the location at which the converted dataset is stored (with filetype `.npy`)

# In[619]:


def convert_data(path="datasets/ratings",cols=(0,1,2),delim="::",dt="int"):
    raw = np.genfromtxt(path+'.dat', usecols=cols, delimiter=delim, dtype=dt)
    np.save(path+".npy",raw)
    # check to see if file works
    assert np.load(path+'.npy').all() == raw.all()
    return path


# The `prep_data` function is used to load the stored data and transform it into a usable and well defined dataframe. 
# 
# In order to do this the function uses the following parameters:
#   * `path` - the (relative) location of the converted dataset: if no file exists a new one is created
#     
# Additionally it returns the following value:
#   * `df_ratings` - a dataframe containing the dataset

# In[620]:


def prep_data(path='datasets/ratings'):
    filepath = path+'.npy'
    if not os.path.isfile(filepath):
        filepath = convert_data()
    ratings = np.load(filepath)
    df_ratings = pd.DataFrame(ratings)
    colnames = ['UserId', 'MovieId', 'Rating']
    df_ratings.columns = colnames
    return df_ratings


# The following snippet is responsible for running the extraction and preparation of the raw data. The data is stored in `df_ratings`.

# In[621]:


df_ratings = prep_data()


# ### **Rating Global**
# 
# The `rating_global` function predicts the user's ratings for a certain movie by taking the mean of all the ratings in the dataset. 
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset
#   * `user` - the user for which a rating is requested [not used, exists for compatibility with generic functions]
#   * `item` - the movie for which a rating is requested [not used, exists for compatibility with generic functions]
#     
# Additionally it returns the following value:
#   * `rating` - the predicted rating for the requested movie by the requested user

# In[629]:


def rating_global(df,user='',item=''):
    rating = df['Rating'].mean()
    return rating


# The following snippet executes a test run of the rating function.

# In[630]:


example = rating_global(df=df_ratings,user=1,item=1193)
print(example)


# ### **Rating Item**
# 
# The `rating_item` function predicts the user's ratings for a certain movie by taking the mean of all the ratings in the dataset for that specific movie.
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset
#   * `item` - the movie for which a rating is requested
#   * `user` - the user for which a rating is requested [not used, exists for compatibility with generic functions]
# 
# Additionally it returns the following value:
#   * `rating` - the predicted rating for the requested movie by the requested user

# In[631]:


def rating_item(df,item,user=''):
    rating = df[df['MovieId']== item].groupby('MovieId')['Rating'].mean()
    return rating[item]


# The following snippet executes a test run of the rating function.

# In[632]:


example = rating_item(df=df_ratings,user=1,item=1193)
print(example)


# ### **Rating User**
# 
# The `rating_user` function predicts the user's ratings for a certain movie by taking the mean of all the ratings in the dataset by the specific user. 
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset
#   * `user` - the user for which a rating is requested
#   * `item` - the movie for which a rating is requested [not used, exists for compatibility with generic functions]
#     
# Additionally it returns the following value:
#   * `rating` - the predicted rating for the requested movie by the requested user

# In[633]:


def rating_user(df,user,item=''):
    rating = df[df['UserId']== user].groupby('UserId')['Rating'].mean()
    return rating[user]


# The following snippet executes a test run of the rating function.

# In[634]:


example = rating_user(df=df_ratings,user=1,item=1193)
print(example)


# ### **Rating User-Item**
# 
# The `rating_user_item` function predicts the user's ratings for a certain movie by applying a linear regression to the outputs of the `rating_user` and `rating_item` functions. 
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset
#   * `user` - the user for which a rating is requested
#   * `item` - the movie for which a rating is requested  
#   * `alpha` - the weight for the `rating_user` function
#   * `beta` - the weight for the `rating_item` function
#   * `gamma` - the offset/modifier for the linear regression
#     
# Additionally it returns the following value:
#   * `rating` - the predicted rating for the requested movie by the requested user    
#   
# Note: `alpha`, `beta` and `gamma` are estimated by the `run_linear_regression` function. 

# In[635]:


def rating_user_item(df,user,item,alpha=0.78212853,beta=0.87673970,gamma=-2.35619748):
    mean_user = rating_user(df,user)
    mean_item = rating_item(df,item)
    rating = alpha * mean_user + beta * mean_item + gamma

    return rating


# The `generate_store_matrix` function is responsible for generating the input matrix for the `run_linear_regression` function. This matrix consists of the user rating, movie rating and a constant term. For easier (re)use the matrix stored into a file.
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset
#   * `path` - the path where the matrix should be stored

# In[162]:


def generate_store_matrix(df,path='datasets/inputLM'):  
    r_user = df.groupby('UserId')['Rating'].mean()
    r_item = df.groupby('MovieId')['Rating'].mean()
    
    matrix = []
    for index, row in df.iterrows():
        matrix.append([index, r_user[row['UserId']], r_item[row['MovieId']], 1])
    np.save(path+'.npy', matrix)  


# The `run_linear_regression` function estimates the values needed for `alpha`, `beta` and `gamma`.
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset
#   * `path` - the location of the input matrix: if no file exists at this location one will be generated
# 
# Additionally it returns the following values:
#   * `alpha` - the estimated weight for the `rating_user` function
#   * `beta` - the estimated weight for the `rating_item` function
#   * `gamma` - the estimated offset/modifier for the linear regression
#   
# Note it is assumed that the indexes in `df` and the matrix stored at `path` correspond, e.g. the data in row 0 of the matrix is computed based on the values of the data in row 0 of `df`.

# In[163]:


def run_linear_regression(df, path='datasets/inputLM'):
    filepath = path+'.npy'
    if not os.path.isfile(filepath):
        generate_store_matrix(df,path)
    matrix = np.load(filepath) 
    y = df['Rating']
    S = np.linalg.lstsq(matrix,y,rcond=None)
    alpha =S[0][0]
    beta = S[0][1]
    gamma = S[0][2]
    print('Sum Squared Error: '+str(S[1]))
    return alpha, beta, gamma


# The following snippet executes a test run of the rating function.

# In[636]:


a,b,g = run_linear_regression(df_ratings)
example = rating_user_item(df=df_ratings,user=1,item=1193,alpha=a,beta=b,gamma=g)
print(example)


# ### **Cross-validation**
# 
# The accuracy of each rating function is computed through 5-fold cross-validation.

# The `run_validation` function executes `n`-fold validation for a given function and initial data set. The initial data is split into `n` test and training folds for which the error is computed. The average error gives an indication of the accuracy of the rating function. 
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the original dataset
#   * `function` a string representing name of the function to be tested
#   * `n` - the number of folds to be generated
#   * `seed` - the random seed to be used
#     
# Additionally it returns the following value:
#   * `train_error` - the average error for this function on the training set
#   * `test_error` - the average error for this function on the test set

# In[272]:


def run_validation(df,function,n=5,seed=17092019):
    err_train = np.zeros(n)
    err_test = np.zeros(n)
    
    kf = KFold(n_splits=n, shuffle=True,random_state=seed)
#     print(kf)

    i = 0
    for train_index, test_index in kf.split(df):
        df_train, df_test = df.iloc[train_index].copy(), df.iloc[test_index].copy()
        
        # run function on training set
        df_train.loc[:,'RatingTrained'] = [function(df=df_train,user=u,item=i) for u, i in zip(df_train['UserId'],df_train['MovieId'])]
#         print(i,'trained')
        # compute error on train set
        df_train.loc[:,'DiffSquared'] = [(t - r)**2 for t, r in zip(df_train['RatingTrained'],df_train['Rating'])]
        err_train[i] = np.sqrt(np.mean(df_train['DiffSquared']))
#         print(i,'train error')
        # compute error on test set
        df_test.loc[:,'DiffSquared'] = [(function(df=df_test,user=u,item=i) - r)**2 for u, i, r in zip(df_test['UserId'],df_test['MovieId'],df_test['Rating'])]
        err_test[i] = np.sqrt(np.mean(df_test['DiffSquared']))  
#         print(i,'test trained and error')
        i = i + 1
    # compute total error
    train_error = np.mean(err_train)
    test_error = np.mean(err_test)
    return train_error, test_error


# The following snippet computes the mean train and test errors for the `rating_global` function.

# In[ ]:


train, test = run_validation(df_ratings,rating_global)
print("Mean training error: " + str(train))
print("Mean test error: " + str(test))


# The following snippet computes the mean train and test errors for the `rating_item` function.

# In[ ]:


train, test = run_validation(df_ratings,rating_item)
print("Mean training error: " + str(train))
print("Mean test error: " + str(test))


# The following snippet computes the mean train and test errors for the `rating_user` function.

# In[ ]:


train, test = run_validation(df_ratings,rating_user)
print("Mean training error: " + str(train))
print("Mean test error: " + str(test))


# The following snippet computes the mean train and test errors for the `rating_user_item` function.

# In[ ]:


train, test = run_validation(df_ratings,rating_user_item)
print("Mean training error: " + str(train))
print("Mean test error: " + str(test))


# ### **Cross-validation Results**
# 
# As can be seen by running the `run_validation` function for the various rating function the performance for each function vastly differs. This is obvious as each function takes a more detailed look (i.e. considers more factors) into what could influence the rating. For a comparison of these results with the model (matrix factorization with gradient descent) please see the `Assignment1_Group15_Model` notebook. 
# 
# Below is an ordered list ranking the different naive functions from most accurate (lowest mean test error) to least accurate:
#   1. `rating_user_item` - test error of XX and training error of YY
#   2. `rating_TODO` - test error of XX and training error of YY 
#   3. `rating_TODO` - test error of XX and training error of YY
#   4. `rating_global` - test error of XX and training error of YY
