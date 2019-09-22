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
# #### **Model Approach**
# This specific notebook handles the implementation of a Matrix factorization approach to the prediction problem.

# The following snippet handles all imports.

# In[2]:


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

# In[3]:


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

# In[4]:


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

# In[5]:


df_ratings = prep_data()


# ### **Rating Model**
# 
# The following functions are used to model and predict user ratings.

# The `generate_model` function computes a model which can be used for predicting a users rating for a requested movie. If no training data can be used estimates are generated using the naive `rating_user_item` approach. Additionally, all ratings are squeezed to be between 1 and 5.
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset on which the model will be computed
#   * `eta` - the learning rate
#   * `lam` - the regularization factor
#   * `max_iter` - the maximum number of iterations allowed in attempting to find a local minimum
#   * `seed` - the random seed to use for generating the initial weights
#   * `alpha` - [for unrated combos] the weight for the average user rating
#   * `beta` -  [for unrated combos] the weight for the average user rating
#   * `gamma` - [for unrated combos] the offset/modifier for the generated prediction
#     
# Additionally it returns the following values:
#   * `model` - a vector containing the predicted rating for each movie
#   * `rmse` - the root-mean-square-error for this model

# In[6]:


def generate_model(df,eta=0.001,lam=0.01,max_iter=1000,seed=22070219,alpha=0.78212853,beta=0.87673970,gamma=-2.35619748):
    # initialize parameters
    matrix = np.nan_to_num(pd.crosstab(df.UserId,df.MovieId,values=df.Rating,aggfunc='mean').to_numpy())
    u = df['UserId'].nunique()
    i = df['MovieId'].nunique()
    np.random.seed = seed
    user = np.random.rand(u,1)
    item = np.random.rand(1,i)
    error = np.full((u, i), np.nan)
    prev_rmse = np.inf
    rmse = np.inf
    curr_rmse = 0
    n = 0
    
    # iterate over all records
    while prev_rmse != curr_rmse:
        it = np.nditer(matrix, flags=['multi_index'],op_flags=['readwrite'])
        while not it.finished:
            rating = it[0]
            if rating is np.nan:
                it.iternext() # skip if no known rating

            a,b = it.multi_index
            u_user = user[a][0]
            i_item = item[0][b]

            err = rating - (u_user * i_item)
            u_user = u_user + eta * ((err * i_item) - (lam * u_user))
            i_item = i_item + eta * ((err * u_user) - (lam * i_item))

            user[a][0] = u_user
            item[0][b] = i_item
            error[a][b] = err
            it.iternext()
        prev_rmse = rmse
        rmse = curr_rmse 
        curr_rmse = np.sqrt(np.nanmean(np.square(error)))
        if n == max_iter:
            break
        n += 1

    model = np.dot(user,item)
    
    # replace random weights of the non-rated user/movie combos with equally weighted average rating for that movie and that user
    item_mean = np.nanmean(model,axis=0)
    user_mean = np.nanmean(model,axis=1)
    inds = np.where(np.isnan(np.where(matrix==0,np.nan,matrix)))
    model[inds] = (alpha*np.take(user_mean,inds[0]) + beta*np.take(item_mean,inds[1]) + gamma)
    
    # ensure ratings are between 1 and 5
    model[model < 1] = 1
    model[model > 5] = 5
    
    return model, rmse


# The `get_best_model` function retreives the model with the lowest rmse for various combinations of `eta` (learning rate) and `lam` (regularization factor).
# 
# In order to do this the function uses the following parameters:
#   * `df` - the dataframe containing the dataset on which the model will be computed
# 
#   * `max_iter` - the maximum number of iterations allowed in attempting to find a local minimum
#   * `max_progression` - the number of different `eta` and `lam` values to be tested (e.g. lenght of the geometric progression)
#   * `progression_ratio` - the ratio to be used when generating the geometric progression
#   
# Additionally it returns the following values:
#   * `model` - a vector containing the best predicted ratings for each user/model.
#   * `rmse` - the root-mean-square-error for the best model
#   * `eta` - the learning rate of the best model
#   * `lam` - the regularization factor of the best model

# In[7]:


def get_best_model(df,max_progression = 4,progression_ratio=3):
    max_iter=1000
    eta_progression = [0.001 * progression_ratio**i for i in range(max_progression)]
    lam_progression = [0.01 * progression_ratio**i for i in range(max_progression)]
    best_model = []
    best_rmse = np.inf
    best_eta = 0
    best_lam = 0
    for i in range(max_progression):
        model, rmse = generate_model(df,eta=eta_progression[i],lam=lam_progression[i],max_iter=max_iter)
        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
            best_eta = eta_progression[i]
            best_lam = lam_progression[i]
        if i != (max_progression - 1):
            model, rmse = generate_model(df,eta=eta_progression[i+1],lam=lam_progression[i],max_iter=max_iter)
            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse
                best_eta = eta_progression[i+1]
                best_lam = lam_progression[i]
            model, rmse = generate_model(df,eta=eta_progression[i],lam=lam_progression[i+1],max_iter=max_iter)
            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse
                best_eta = eta_progression[i]
                best_lam = lam_progression[i+1]
    return best_model, best_rmse, best_eta, best_lam


# The following snippet retrieves the best model:

# In[8]:


X = df_ratings #.iloc[:200]
model, rmse, eta, lam = get_best_model(X)#\
#model, rmse, eta, lam = get_best_model(df_ratings)
np.savez('datasets/model-rmse-eta-lam', model, rmse, eta, lam)
print('The best model has rmse of '+str(rmse)+' and eta of '+str(eta)+' and lam of '+str(lam))


# The `rating_model` function predicts the user's ratings for a certain movie by implenting Matrix factorization with Gradient Descent. 
# 
# In order to do this the function uses the following parameters:
#   * `model` - the model to be used to retrieve the ratings
#   * `df` - the dataframe containing the dataset
#   * `user` - the user for which a rating is requested
#   * `item` - the movie for which a rating is requested 
#     
# Additionally it returns the following value:
#   * `rating` - the predicted rating for the requested movie by the requested user

# In[9]:


def rating_model(model,df,user,item):
    users = df['UserId'].unique()
    u = np.where(users == user)[0][0]
    items = df['MovieId'].unique()
    i = np.where(items == item)[0][0]
    rating = model[u][i] 
    return rating


# The following snippet executes a test run of the rating function.

# In[10]:


a = rating_model(model,X,1,1193)
print(a)


# ### **Cross-validation**
# 
# The accuracy of the model is computed through 5-fold cross-validation.

# The `run_validation` function executes `n`-fold validation for a given function and initial data set. The initial data is split into `n` test and training folds for which the error is computed. The average error gives an indication of the accuracy of the rating function. 
# 
# In order to do this the function uses the following parameters:
#   * `model` - the model to be used to retrieve the ratings
#   * `df` - the dataframe containing the original dataset
#   * `n` - the number of folds to be generated
#   * `seed` - the random seed to be used
#     
# Additionally it returns the following value:
#   * `train_error` - the average error for this function on the training set
#   * `test_error` - the average error for this function on the test set

# In[11]:


def run_validation(m,df,n=5,seed=17092019):
    err_train = np.zeros(n)
    err_test = np.zeros(n)
    print(df)
    kf = KFold(n_splits=n, shuffle=True,random_state=seed)

    i = 0
    for train_index, test_index in kf.split(df):
        df_train, df_test = df.iloc[train_index].copy(), df.iloc[test_index].copy()
        
        # run function on training set
        df_train.loc[:,'RatingTrained'] = [rating_model(model=m,df=df_train,user=u,item=i) for u, i in zip(df_train['UserId'],df_train['MovieId'])]
#         print(i,'trained')
        # compute error on train set
        df_train.loc[:,'DiffSquared'] = [(t - r)**2 for t, r in zip(df_train['RatingTrained'],df_train['Rating'])]
        err_train[i] = np.sqrt(np.mean(df_train['DiffSquared']))
#         print(i,'train error')
        # compute error on test set
        df_test.loc[:,'DiffSquared'] = [(rating_model(model=m,df=df_test,user=u,item=i) - r)**2 for u, i, r in zip(df_test['UserId'],df_test['MovieId'],df_test['Rating'])]
        err_test[i] = np.sqrt(np.mean(df_test['DiffSquared']))  
#         print(i,'test trained and error')
        i = i + 1
    # compute total error
    train_error = np.mean(err_train)
    test_error = np.mean(err_test)
    return train_error, test_error


# The following snippet computes the mean train and test errors for the `rating_model`

# In[14]:


train, test = run_validation(model,X)
#train, test = run_validation(model,df_ratings)
np.savez('datasets/train_test_model', train, test)
print("Mean training error: " + str(train))
print("Mean test error: " + str(test))


# ### **Cross-validation Results**
# 
# As can be seen by running the `run_validation` function for the various rating function the performance for each function vastly differs. This is obvious as each function takes a more detailed look (i.e. considers more factors) into what could influence the rating. TODO TODO TODO TODO
# 
# Below is an ordered list ranking the different functions from most accurate (lowest mean test error) to least accurate:
#   1. `rating_model` - test error of XX and training error of YY
#   2. `rating_user_item` - test error of XX and training error of YY
#   3. `rating_TODO` - test error of XX and training error of YY 
#   4. `rating_TODO` - test error of XX and training error of YY
#   5. `rating_global` - test error of XX and training error of YY
