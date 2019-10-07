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

# In[1]:


import sys
import numpy as np
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import multiprocessing
import timeit


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

# In[ ]:


# user_movie = np.load('datasets/user_movie.npy')
# user_movie = user_movie[:1000,]


# In[6]:


# %%time
user_movie = np.load('datasets/user_movie.npy')
c = user_movie[:,0]
r = user_movie[:,1]
d = np.ones(len(c))
max_c = len(np.unique(c))
max_r = len(np.unique(r))
m = csr_matrix((d, (r,c)), shape=(max_r, max_c))
signature_length = 10
hash_func = [np.random.permutation(max_r) for i in range(signature_length)]
matrix = np.full((signature_length, max_c), np.inf)


# In[ ]:


# what are your inputs, and what operation do you want to 
# perform on each input. For example...
def processInput(i,j):
    # print(i,j)
    for h_i in range(signature_length):
        hash = hash_func[h_i]
        # print(hash[i], (i,j), matrix[h_i][j])
        if(hash[i] < matrix[h_i][j]):
            matrix[h_i][j] = hash[i]
        # print(hash)
        
        # print(hash[i], (i,j), matrix[h_i][j])
        # if(timeit.default_timer()-start > 30):
        #     print('STOP')
        #     break
            
    np.save('datasets/sign_matrix', matrix)

            


# In[ ]:


def main(argv):
    seed = sys.argv[1]
    path = sys.argv[2]
    print(seed, path)
    
    num_cores = 8
    print(num_cores)
    
    start = timeit.default_timer()
    Parallel(n_jobs=num_cores, max_nbytes='50M')(delayed(processInput)(i,j) for i,j in zip(*m.nonzero()))
    end = timeit.default_timer() - start
    
    print('Done in: ', end, 'seconds!')
    


# The following snippet passes the start of the program and the command line arguments to the `main` function.

# In[ ]:


if __name__ == "__main__":
    main(sys.argv[1:])

