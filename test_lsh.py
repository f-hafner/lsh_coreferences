
import numpy as np 
import lsh 


def test_cols_to_int():
    a = np.array([[1,4,10], [14, 12, 3], [1, 100, 39]])
    a_reduced = lsh.cols_to_int(a).squeeze()
    expected = np.array([[1410, 14123, 110039]])
    assert np.all(a_reduced == expected), f"cols_to_int fails for {a}"


def test_idx_unique_multidim():   
    inputs = [
        np.array([[1, 3], [2, 2], [2, 2], [1, 3], [1, 5], [1, 1]]),
        np.array([[3,4], [3,5], [5,6], [3,4], [6,7]]) 
    ]
    expected = [
        [np.array([5]), np.array([0, 3]), np.array([4]), np.array([1, 2])],
        [np.array([0, 3]), np.array([1]), np.array([2]), np.array([4])]
    ]
    for input, expected_output in zip(inputs, expected):
        output = lsh.idx_unique_multidim(input)
        assert len(expected_output) == len(output), "different lengths"
        compare = [np.all(e == o) for e, o in zip(expected_output, output)]
        assert all(compare), f"different groups for \n {input}"
