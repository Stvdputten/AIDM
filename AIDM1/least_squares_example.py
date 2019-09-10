# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:30:51 2017

@author: Wojtek
"""
import numpy as np

#generate some artificial data: 
x1=np.random.rand(10)
x2=np.random.rand(10)

y=2*x1 + 3*x2 + 0.5 + 0.1*np.random.randn(10)

#let's assume that we are given only x1, x2 and y
#Our task is to find coefficients A, B, C, such that the linear combination:
#A*x + B*y + C  approximates y as good as possible.

#We can do it as follows:
#Create a matrix X with three columns: [x1, x2, np.ones(10)]:    

X=np.vstack([x1, x2, np.ones(10)]).T

#vstack creates a matrix with 3 rows; .T transposes it - exactly what we want

#Now we can find coefficents A, B, C by:
    
S = np.linalg.lstsq(X,y)    

#S is a list of several objects: 
#S[0] is what we need (coefficients, A, B, C):

print(S[0])

#S[1] is the Sum Squared Error:
print(S[1])
    
# S[2] and S[3] which are, at this moment, irrelevant.
############
    
#Alternatively, you can use a sklearn package
#http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
# to get the same result:

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit (X[:,0:2],y)

print(reg.coef_)
print(reg.intercept_)

#Note that we ignored the last column of X that consists of 1's. 
#By default, linear.model.LinearRegression assumes that is has to find an intercept so we don't need this last colum.
