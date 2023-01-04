
import numpy as np 

import lsh 



def test_cols_to_int():
    a = np.array([[1,4,10], [14, 12, 3], [1, 100, 39]])
    a_reduced = lsh.cols_to_int(a).squeeze()
    expected = np.array([[1410, 14123, 110039]])
    assert np.all(a_reduced == expected), f"cols_to_int fails for {a}"


def test_idx_unique_multidim():   
    pass # finish here 


x1 = np.array([[1, 3], [2, 2], [2, 2], [1, 3], [1, 5], [1, 1]]) # this is one test case
x2 = np.array([[3,4], [3,5], [5,6], [3,4], [6,7]]) 
x3 = np.array([[1,4,10], [14, 12, 3], [1, 100, 39], [14, 12, 3]])


display(x1)
display(idx_unique_final(x1))

display(x2)
display(idx_unique_final(x2))

display(x3)
display(idx_unique_final(x3))