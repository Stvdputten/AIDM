#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## **Advances in Data Mining**
# 
# Stephan van der Putten | (s1528459) | stvdputtenjur@gmail.com  
# Theo Baart | s2370328 | s2370328@student.leidenuniv.nl
# 
# ### **Assignment 2**
# This assignment is concerned with finding the set of similar users in the provided datasource. To be more explicit, in finding all pairs of users who have a Jaccard similarity of more than 0.5. Additionally, this assignment considers comparing the "naïve implementation" with the "LSH implementation". The "naïve implementation" can be found in the file `time_estimate.ipynb` and the "LSH implementation" in the file `lsh.ipynb`.
# 
# Note all implementations are based on the assignment guidelines and helper files given as well as the documentation of the used functions. 
# 
# #### **LSH Implementation**
# This notebook implements LSH in order to find all pairs of users with a Jaccard similarity of more than 0.5. As noted in the assignment instructions the data file is loaded from `user_movie.npy` and the list of user pairs are printed in the file `ans.txt`. Additionally, this implementation supports the setting of a random seed to determine the permutations to be used in LSH. The algorithm will continually save its output so as to aid in the evluation criteria which only looks at the first 15 minutes of the LSH execution.
# ___

# The following snippet handles all imports.

# In[2]:


import time
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix, find
from scipy.sparse import identity


# ### **Program Execution**
# This section is concerned with parsing the input arguments and determining the execution flow of the program.
# 
# ___
# The `main` function handles the start of execution from the command line.
# 
# In order to do this the function uses the following parameters:
#   * `argv` - the command line arguments given to the program
#   
# The following command line arguments are expected:
#   * `seed` - the value to use as random seed
#   * `path` - the location of the `user_movies.npy` file

# In[3]:


user_movie = np.load('datasets/user_movie.npy')


# In[4]:


get_ipython().run_cell_magic('time', '', 'c = user_movie[:,0]\nr = user_movie[:,1]\nd = np.ones(len(c))\nmax_c = len(np.unique(c))\nmax_r = len(np.unique(r))\n# m = csr_matrix((d, (r,c)), shape=(max_r, max_c))\ncsc = csc_matrix((d, (r,c)), shape=(max_r, max_c))\ncsr = csr_matrix((d, (r,c)), shape=(max_r, max_c))\nsignature_length = 50\n\n# example = np.array([[1,0,0,1],[0,0,1,0],[0,1,0,1],[1,0,1,0],[0,0,1,0]])\n# hash_func = np.array([[4,3,1,2,0], [3,0,4,2,1]])')


# In[47]:


def rowminhash(signature_length, hashfunc, matrix):
    sigm = np.full((signature_length, matrix.shape[1]), np.inf)
    for row in range(matrix.shape[0]):
        ones = find(matrix[row, :])[1]
        hash = hash_func[:,row]
        B = sigm.copy()
        B[:,ones] = 1
        B[:,ones] = np.multiply(B[:,ones], hash.reshape((len(hash), 1)))
        # B[:, ones] *= hash.reshape((len(hash),1))
        sigm = np.minimum(sigm, B)
        # print(example[row%len(hash_func), ones] * hash.reshape((len(hash),1)))
        # np.amin()
        # sigm[example]
        # sigm[]
        
        # print(example[row%len(hash_func), ones]*hash.reshape((len(hash),1)))
        # np.dot(example[row%len(hash_func)],hash)
        # row_sign = np.amin(hash, axis=0)
        # print(np.multiply(np.array([[1,1],[1,1]]), np.array([[2],[3]])))
        # print('Rowsgn = ', row_sign)
        # sigm[row%len(hash_func), ones] = hash
        # print('Row =', row, 'Ones =' ,ones, 'Hash =', hash)
        # print(sigm[row%len(hash_func), ones])
        # for row,col in zip(*example.nonzero()):
        # hash = hash_func[:, row]
        # print('Row =', row, 'Ones =' ,ones, 'Hash =', hash)
        # print(row,col)
    return(sigm)


# In[19]:


def minhash(signature_length,hashfunc, matrix):
    # t0 = time.time()
    sigm = np.full((signature_length, matrix.shape[1]), np.inf)
    # print(sigm)
    # hash_func = np.array([np.random.permutation(matrix.shape[0]) for i in range(signature_length)])
    # print(hash_func)
    for r,c in zip(*matrix.nonzero()):
        # print('r = ', r, 'c = ', c)
        for h_i in range(signature_length):
            # print('h_i = ' ,h_i)
            hash = hash_func[h_i]
            if(hash[r] < sigm[h_i][c]):
                # print(hash[r])
                sigm[h_i][c] = hash[r]
            # print("\nGenerating MinHash signatures took %.2fsec" % elapsed)
        # elapsed = (time.time() - t0)            
    return(sigm)  


# In[ ]:


def do_lsh(sign_matrix, signature_length, threshold):
    return 0


# In[48]:


np.random.seed = 42

# example = csr
# example = np.array([[1,0,0,1],[0,0,1,0],[0,1,0,1],[1,0,1,0],[0,0,1,0]])
hash_func = np.array([np.random.permutation(csr.shape[0]) for i in range(signature_length)])
# %time sigm1 = minhash(signature_length,hash_func, example)
get_ipython().run_line_magic('time', 'sigm1 = rowminhash(100 ,hash_func, csr)')
get_ipython().run_line_magic('time', 'sigm2 = rowminhash(signature_length,hash_func, csr)')
# print(sigm2)
np.save('datasets/sign_matrix_100', sigm1)
np.save('datasets/sign_matrix', sigm2)

# print(np.equal(sigm1, sigm2))


# In[ ]:


print(hash_func)
# csr1 = example
for row in range(csr1.shape[1]):
    ones = find(csr1[row,:])[1]
    # print(ones)
    hash = hash_func[:, ones]
    print(hash)
    # np.amin()
    
    # row_signature = np.amin(hash,).reshape((1,signature_length))
# 
# for row in range(test[:,:2].shape[1]):
#     ones_index = np.where(test[row,:]==1)
# for row in range(n_row):
#     ones_index = np.where(u[row,:]==1)[0]
#     corresponding_hashes = hash_code[:,ones_index]
# 
#     row_signature = np.amin(corresponding_hashes,axis=1).reshape((1,num_of_hashes))
#     
#     signature_array[row,:] = row_signature
#     
#     if row % 10000 == 0 :
#       #print(row)
#       print (str(round(row*100/n_row,2))+' percent complete in '+str(round(time.time()-t2,2))+' seconds')


# In[ ]:


def main(argv):
    seed = sys.argv[1]
    path = sys.argv[2]
    print(seed, path)


# The following snippet passes the start of the program and the command line arguments to the `main` function.

# In[ ]:


if __name__ == "__main__":
    main(sys.argv[1:])

