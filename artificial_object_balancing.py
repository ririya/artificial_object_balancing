"""
Author: Rafael Iriya
Nov 2020
lsqlin code by https://github.com/visva89/
"""

import numpy as np
from cvxopt import solvers, matrix, spmatrix, mul
import itertools
from scipy import sparse
import random
import bisect


def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
    return SP


def spmatrix_sparse_to_scipy(A):
    data = np.array(A.V).squeeze()
    rows = np.array(A.I).squeeze()
    cols = np.array(A.J).squeeze()
    return sparse.coo_matrix((data, (rows, cols)))


def sparse_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return sparse.vstack([A1, A2])


def numpy_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.vstack([A1, A2])


def numpy_None_concatenate(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.concatenate([A1, A2])


# def get_shape(A):
def get_shape(C):
    if isinstance(C, spmatrix):
        return C.size
    else:
        return C.shape


def numpy_to_cvxopt_matrix(A):
    if A is None:
        return A
    if sparse.issparse(A):
        if isinstance(A, sparse.spmatrix):
            return scipy_sparse_to_spmatrix(A)
        else:
            return A
    else:
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                return matrix(A, (A.shape[0], 1), 'd')
            else:
                return matrix(A, A.shape, 'd')
        else:
            return A


def cvxopt_to_numpy_matrix(A):
    if A is None:
        return A
    if isinstance(A, spmatrix):
        return spmatrix_sparse_to_scipy(A)
    elif isinstance(A, matrix):
        return np.array(A).squeeze()
    else:
        return np.array(A).squeeze()


def lsqlin(C, d, reg=0, A=None, b=None, Aeq=None, beq=None, \
           lb=None, ub=None, x0=None, opts=None):
    '''
        Solve linear constrained l2-regularized least squares. Can
        handle both dense and sparse matrices. Matlab's lsqlin
        equivalent. It is actually wrapper around CVXOPT QP solver.

            min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
            s.t.  A * x <= b
                  Aeq * x = beq
                  lb <= x <= ub

        Input arguments:
            C   is m x n dense or sparse matrix
            d   is n x 1 dense matrix
            reg is regularization parameter
            A   is p x n dense or sparse matrix
            b   is p x 1 dense matrix
            Aeq is q x n dense or sparse matrix
            beq is q x 1 dense matrix
            lb  is n x 1 matrix or scalar
            ub  is n x 1 matrix or scalar

        Output arguments:
            Return dictionary, the output of CVXOPT QP.

        Dont pass matlab-like empty lists to avoid setting parameters,
        just use None:
            lsqlin(C, d, 0.05, None, None, Aeq, beq) #Correct
            lsqlin(C, d, 0.05, [], [], Aeq, beq) #Wrong!
    '''
    sparse_case = False
    if sparse.issparse(A):  # detects both np and cxopt sparse
        sparse_case = True
        # We need A to be scipy sparse, as I couldn't find how
        # CVXOPT spmatrix can be vstacked
        if isinstance(A, spmatrix):
            A = spmatrix_sparse_to_scipy(A)

    C = numpy_to_cvxopt_matrix(C)
    d = numpy_to_cvxopt_matrix(d)
    Q = C.T * C
    q = - d.T * C
    nvars = C.size[1]

    if reg > 0:
        if sparse_case:
            I = scipy_sparse_to_spmatrix(sparse.eye(nvars, nvars, \
                                                    format='coo'))
        else:
            I = matrix(np.eye(nvars), (nvars, nvars), 'd')
        Q = Q + reg * I

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b = cvxopt_to_numpy_matrix(b)

    if lb is not None:  # Modify 'A' and 'b' to add lb inequalities
        if lb.size == 1:
            lb = np.repeat(lb, nvars)

        if sparse_case:
            lb_A = -sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, lb_A)
        else:
            lb_A = -np.eye(nvars)
            A = numpy_None_vstack(A, lb_A)
        b = numpy_None_concatenate(b, -lb)
    if ub is not None:  # Modify 'A' and 'b' to add ub inequalities
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if sparse_case:
            ub_A = sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, ub_A)
        else:
            ub_A = np.eye(nvars)
            A = numpy_None_vstack(A, ub_A)
        b = numpy_None_concatenate(b, ub)

    # Convert data to CVXOPT format
    A = numpy_to_cvxopt_matrix(A)
    Aeq = numpy_to_cvxopt_matrix(Aeq)
    b = numpy_to_cvxopt_matrix(b)
    beq = numpy_to_cvxopt_matrix(beq)

    # Set up options
    if opts is not None:
        for k, v in opts.items():
            solvers.options[k] = v

    # Run CVXOPT.SQP solver
    sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, x0)
    return sol


def lsqnonneg(C, d, opts):
    '''
    Solves nonnegative linear least-squares problem:

    min_x ||C*x - d||_2^2,  where x >= 0
    '''
    return lsqlin(C, d, reg=0, A=None, b=None, Aeq=None, \
                  beq=None, lb=0, ub=None, x0=None, opts=opts)


def get_number_artificial_objects_LS(orig_freq):

    orig_total = np.sum(orig_freq)
    n_classes = len(orig_freq)

    C = []
    d = []

    for ind in range(n_classes):
        curr_row = [1] * n_classes
        curr_row[ind] = 1 - n_classes
        C.append(curr_row)
        d.append(n_classes * orig_freq[ind] - orig_total)

    C = np.array(C)
    d = np.array(d)

    x = np.dot(np.linalg.pinv(C), d)

    new_total = orig_total + np.sum(x)

    new_probs = []

    for ind in range(n_classes):
        new_probs.append((orig_freq[ind] + x[ind]) / new_total)

    prob_addition = [abs(xx) / np.sum(abs(x)) for xx in x]

    sign_addition = [1 if xx > 0 else -1 for xx in x]

    print('least squares solution:')
    print(str.format("new_probs = {}", new_probs))
    print(str.format("x = {}", x))

    return x, prob_addition, sign_addition


def get_number_artificial_objects_min_norm(orig_freq):
    orig_total = np.sum(orig_freq)
    n_classes = len(orig_freq)

    C = []
    d = []

    for ind in range(n_classes):
        curr_row = [1] * n_classes
        curr_row[ind] = 1 - n_classes
        C.append(curr_row)
        d.append(n_classes * orig_freq[ind] - orig_total)

    C = np.array(C)
    d = np.array(d)

    ret = lsqlin(C, d, reg=1)

    x = np.round(ret['x'].T)[0]

    new_total = orig_total + np.sum(x)

    new_probs = []

    for ind in range(n_classes):
        new_probs.append((orig_freq[ind] + x[ind]) / new_total)

    prob_addition = [abs(xx) / np.sum(abs(x)) for xx in x]

    sign_addition = [1 if xx > 0 else -1 for xx in x]

    print('nonnegative solution:')
    print(str.format("new_probs = {}", new_probs))
    print(str.format("x = {}", x))
    print(str.format("prob_addition = {}", prob_addition))

    return x, prob_addition, sign_addition

def get_number_artificial_objects_nonnegative(orig_freq):
    orig_total = np.sum(orig_freq)
    n_classes = len(orig_freq)

    C = []
    d = []

    for ind in range(n_classes):
        curr_row = [1] * n_classes
        curr_row[ind] = 1 - n_classes
        C.append(curr_row)
        d.append(n_classes * orig_freq[ind] - orig_total)

    C = np.array(C)
    d = np.array(d)

    ret = lsqnonneg(C, d, {'show_progress': False})

    x = np.round(ret['x'].T)[0]

    new_total = orig_total + np.sum(x)

    new_probs = []

    for ind in range(n_classes):
        new_probs.append((orig_freq[ind] + x[ind]) / new_total)

    prob_addition = [xx / np.sum(x) for xx in x]
    sign_addition = [1 if xx > 0 else -1 for xx in x]

    print('nonnegative solution:')
    print(str.format("new_probs = {}", new_probs))
    print(str.format("x = {}", x))
    print(str.format("prob_addition = {}", prob_addition))

    return x, prob_addition, sign_addition

##MAIN
if __name__ == '__main__':

    mode = "MINNORM"
    orig_freq = np.array([10,2,3,1,4,8])*100
    orig_total = np.sum(orig_freq)
    n_classes = len(orig_freq)

    orig_probs = [f / np.sum(orig_freq) for f in orig_freq]

    orig_probs_cum = []
    curr_sum = 0
    for p in orig_probs:
        curr_sum += p
        orig_probs_cum.append(curr_sum)

    if mode == "LSQ":
        x, prob_addition, sign_addition = get_number_artificial_objects_LS(orig_freq)     # LEAST SQUARES
    # SOLUTION

    if mode == "NONNEG":
        x, prob_addition, sign_addition = get_number_artificial_objects_nonnegative(orig_freq) #NONNEGATIVE SOLUTION

    if mode == "SIMPLE":
        x = np.zeros(len(orig_freq,))
        for ind, f in enumerate(orig_freq):
            x[ind] = max(orig_freq) - f
        prob_addition = [xx / np.sum(x) for xx in x]
        sign_addition = [1]*len(orig_freq)

    if mode == "MINNORM":
        x, prob_addition, sign_addition = get_number_artificial_objects_min_norm(orig_freq)

    print(x)

    #TESTING ALGORITHM
    modified_freq = np.array([0]*n_classes)

    probs_cum = []
    curr_sum = 0

    for p in prob_addition:
        curr_sum += p
        probs_cum.append(curr_sum)

    num_images = 1000

    num_extra_additions = 0

    number_additions_per_image = int(np.round(np.sum(abs(x))/num_images))
    decimal = np.sum(abs(x)) / num_images - np.floor(np.sum(abs(x)) / num_images)
    try:
        extra_addition = int(round(1/decimal)*num_extra_additions)
    except:
        extra_addition = 0

    average_number_objects_per_image = int(np.round(np.sum(orig_freq)/num_images))
    decimal = np.sum(abs(orig_freq)) / num_images - np.floor(np.sum(abs(orig_freq)) / num_images)
    try:
        extra_object = int(round(1 / decimal)*num_extra_additions)
    except:
        extra_object = 0

    cum_to_modify = [0]*n_classes

    for n in range(num_images*100):

        curr_to_modify = [0] * n_classes

        img_freq = [0]*n_classes

        if extra_object:
            number_objects = average_number_objects_per_image + num_extra_additions if n % extra_object == \
                                                                 0 else average_number_objects_per_image
        else:
            number_objects = average_number_objects_per_image

        for _ in range(number_objects):

            rand = random.random()

            ind = bisect.bisect_left(orig_probs_cum, rand)
            img_freq[ind] += 1

        if extra_addition:
            number_additions = number_additions_per_image + num_extra_additions if n % extra_addition == 0 else \
                number_additions_per_image
        else:
            number_additions = number_additions_per_image

        for _ in range(number_additions):

            rand = random.random()

            ind = bisect.bisect_left(probs_cum, rand)
            curr_to_modify[ind]+=1

        for ind_mod in range(n_classes):

            while curr_to_modify[ind_mod] > 0 and img_freq[ind_mod] + sign_addition[ind_mod] >= 0:
                    img_freq[ind_mod] += sign_addition[ind_mod]
                    curr_to_modify[ind_mod] -= 1

            modified_freq[ind_mod] += img_freq[ind_mod]

            cum_to_modify[ind_mod] += curr_to_modify[ind_mod]

        for ind_mod in range(n_classes):

            while cum_to_modify[ind_mod] > 0 and img_freq[ind_mod] + sign_addition[ind_mod] >= 0:
                img_freq[ind_mod] += sign_addition[ind_mod]
                cum_to_modify[ind_mod] -= 1

            modified_freq[ind_mod] += img_freq[ind_mod]

        if n%100 == 0:
            modified_probs = [m / np.sum(modified_freq) for m in modified_freq]
            print(str.format("modified_probs = {}", modified_probs))
            print(str.format("modified_freq = {}", modified_freq))

    print(str.format("modified_probs = {}", modified_probs))
    print(str.format("modified_freq = {}", modified_freq))

    rmse = np.linalg.norm(np.array(modified_probs) - 1/n_classes)

    print(str.format("rmse = {}", rmse*100))









