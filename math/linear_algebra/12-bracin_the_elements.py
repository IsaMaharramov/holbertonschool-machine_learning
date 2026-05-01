#!/usr/bin/env python3
"""
Defines a function that performs element-wise operations on numpy.ndarrays
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    Args:
        mat1: first numpy.ndarray or scalar
        mat2: second numpy.ndarray or scalar
    Returns:
        A tuple containing the element-wise sum, difference, product,
        and quotient
    """
    sum_res = mat1 + mat2
    diff_res = mat1 - mat2
    prod_res = mat1 * mat2
    quot_res = mat1 / mat2

    return (sum_res, diff_res, prod_res, quot_res)
