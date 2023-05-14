import os
import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle

# Using latex for the axis of the plot
# plt.rcParams['text.usetex'] = True

def cal_sum_rank_1(A):
    """
    Calculate the first part of the smoothness matrix (1/4 sum) and the threshold lambda
    Input:  dataset A
    Output: the first part of smoothness matrix, the minimum, maximum eigenvalue of it
    """
    n, d = A.shape
    sum = np.zeros((d, d))
    for i in range(n):
        A_i_col, A_i_row = A[i][:, np.newaxis], A[i][np.newaxis, :]
        sum += A_i_col @ A_i_row
    sum = (1 / (4 * n)) * sum
    lam_min = np.abs(np.min(sp.linalg.eigvals(sum)))
    lam_max = np.abs(np.max(sp.linalg.eigvals(sum)))
    # Return the sum of the rank one matrices with cofficient, and the minimum eigenvalue
    return [sum, lam_min, lam_max]

def sigmoid(x):
    """
    Sigmoid function used in the calculation
    Input:  a scalar x
    Output: the result of sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def cal_matrix_norm(D, x):
    """
    Calculate matrix norm for the current iterate
    Input:  a PSD matrix D, and a vector x of corresponging length
    Output: square matrix norm of the vector
    """
    return (x[np.newaxis, :] @ D @ x[:, np.newaxis])[0][0]

def plot_eigs(L):
    """
    Plot the eigenvalue distribution of the smoothness matrix L
    Input: the smoothness matrix L
    """
    eig_vals_L, eig_vecs_L = np.linalg.eigh(L)
    num_bins = int(len(eig_vals_L)) * 2
    plt.hist(eig_vals_L, bins=num_bins)
    plt.xlabel("Eigenvalues of smoothness matrix")
    plt.ylabel("Number of eigenvalues")
    plt.show()

def cal_grad(A, b, x, lam):
    """
    Calculate the gradient of the for logistic regression with non-convex regularizer
    Input:  A is the dataset, b is the corresponding labels, x is the iterate, lam is the value of lambda
    Output: the gradient of function with non-convex regularizer in this case
    """
    n, d = A.shape
    # Calculating grad of g(x):
    inner_mat = A * x
    inner_slr = np.sum(inner_mat, axis=1)
    inner_slr_b = -b * inner_slr
    inner_sig = sigmoid(inner_slr_b)
    inner_sig_b = -b * inner_sig
    inner_grad_mat = inner_mat * inner_sig_b[:, np.newaxis]
    grad_g = np.average(inner_grad_mat, axis=0)
    # Calculating grad of r(x):
    numer = 2 * x
    denom = (1 + x ** 2) ** 2
    grad_r = numer / denom
    return grad_g + lam * grad_r

def rand_t_sparsifier(x, t):
    """
    Random t sparsifier
    Input:  vector x, the minibatch size t
    Output: vector x after rand-t sparsifier
    """
    d = x.shape[0]
    if t > d:
        raise AssertionError("Exceeding minibatch")
    mask = np.zeros_like(x)
    idx = np.random.choice(np.arange(d), size=t, replace=False)
    mask[idx] = d / t
    return mask * x

def special_rand_1(x, idx):
    """
    Specially designed rand-1 sparisifer to guarantee same randomness
    Input:  x is the vector, idx is a list of indices specifyinng the coordinates picked
    Output: x after applying rand-1 sparidifer
    """
    mask = np.zeros_like(x)
    mask[idx] = d
    return mask * x

def cal_func_val(A, b, x, lam):
    """
    Calculate the function value
    Input:  A is the dataset, b is the corresponding label, x is the vector where we are evaluating
            the function value, lam is the value of lambda we used for the non-convex regularizer
    Output: the function value at vector x
    """
    # Function value of g part
    temp_1 = np.sum(A * x, axis=1)
    temp_2 = temp_1 * (-b)
    temp_3 = np.log(np.exp(temp_2) + 1)
    g_value = np.average(temp_3)
    # Function value of r part
    numerator = x ** 2
    denominator = 1 + x ** 2
    r_value = np.sum(numerator / denominator)
    return g_value + lam * r_value

# Functions for experiment 1, results can be found in folder './result_exp_1'
def run_cgd_curve_1_exp_1(x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with scalar stepsize, scalar smoothness constant, rand-1 sketch is used
    The average norm is solved as "logistic_exp_1_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_0     = (1 / (d * L_scalar))
    D           = np.eye(d) * gamma_0
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_0 * special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 1", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_1_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_2_exp_1(x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with scalar stepsize, matrix smoothness, rand-1
    The average norm is solved as "logistic_exp_2_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_1     = (1 / (d * L_scalar))
    D           = np.eye(d) * gamma_1
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_1 * special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 2", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_2_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_3_exp_1(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with diagonal matrix stepsize, matrix smoothness, rand-1, for algorithm 1
    The average norm is solved as "logistic_exp_1_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_2     = 1 / d
    D           = L_matrix * gamma_2
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - D @ special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_3_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_4_exp_1(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with diagonal matrix stepsize, matrix smoothness, rand-1, for algorithm 2
    The average norm is solved as "logistic_exp_1_curve_4_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_3     = 1 / d
    D           = L_matrix * gamma_3
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - special_rand_1(D @ grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 4", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_4_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_gd_scalar(x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running gradient descent with scalar stepsize, scalar smoothness / matrix smoothness (These two are the same)
    The average norm is solved as "logistic_exp_1_curve_5_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_5 = 1 / L_scalar
    D = np.eye(d) * gamma_5
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_5 * grad_x
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 5", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_5_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_gd_matrix(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running gradient descent with matrix stepsize, matrix smoothness
    The average norm is solved as "logistic_exp_1_curve_6_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    D = L_matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - D @ grad_x
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 6", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_6_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_7_exp_1(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with L-1 matrix stepsize, matrix smoothness, rand-1, for algorithm 1
    The average norm is solved as "logistic_exp_1_curve_7_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get L related matrix
    L_half      = sp.linalg.fractional_matrix_power(L_matrix, 1/2)
    L_inv       = sp.linalg.inv(L_matrix)
    L_inv_diag  = np.diag(np.diag(L_inv))
    # Get stepsize matrix
    gamma_7     = (1 / (np.linalg.eigh(L_half @ L_inv_diag @ L_half)[0][-1])) * (1/d)
    D           = L_inv * gamma_7
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - D @ special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 7", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_7_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_8_exp_1(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with L-1 matrix stepsize, matrix smoothness, rand-1, for algorithm 1
    The average norm is solved as "logistic_exp_1_curve_8_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get L related matrix
    L_half      = sp.linalg.fractional_matrix_power(L_matrix, 1/2)
    L_half_inv  = sp.linalg.fractional_matrix_power(L_matrix, -(1/2))
    # Get stepsize matrix
    gamma_8     = (1 / (np.linalg.eigh(L_half)[0][-1])) * (1/d)
    D           = L_half_inv * gamma_8
    D_1d        = sp.linalg.fractional_matrix_power(D, 1/d)
    D_1d_det    = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - D @ special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 1 curve 8", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_1_curve_8_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_1", result_file)
    np.save(outfile, arrnm)
    return arrnm


# Functions for experiment 2, results can be found in folder './result_exp_2'
def run_cgd_curve_1_exp_2(x_iter, L_scalar, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running CGD with scalar stepsize, scalar smoothness constant, rand-t
    The average norm is solved as "logistic_exp_2_curve_1_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Get stepsize matrix
    gamma_1         = (t / d) * (1 / L_scalar)
    D               = np.eye(d) * gamma_1
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_1 * rand_t_sparsifier(grad_x, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 1", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_1_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_2_exp_2(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running CGD with scalar stepsize, matrix smoothness, rand-t
    The average norm is solved as "logistic_exp_2_curve_2_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L))
    L_comb = ((d - t)/(d - 1)) * L_diag + ((t - 1)/(d - 1)) * L
    max_eig_L_comb = np.linalg.eigh(L_comb)[0][-1]
    # Get stepsize matrix
    gamma_2         = (t / d) * (1 / max_eig_L_comb)
    D = np.eye(d) * gamma_2
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_2 * rand_t_sparsifier(grad_x, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 2", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_2_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_3_exp_2(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running algorithm 2 with optimal matrix stepsize, matrix smoothness, rand-t
    The average norm is solved as "logistic_exp_2_curve_3_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L))
    L_comb = (d / t) * (((d - t) / (d - 1)) * L_diag + ((t - 1) / (d - 1)) * L)
    # Get stepsize matrix
    D =  np.linalg.inv(L_comb)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - rand_t_sparsifier(D @ grad_x, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_3_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_4_exp_2(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null', t=0):
    """
    Running algorithm 1 with matrix stepsize, matrix smoothness, rand-t
    The average norm is solved as "logistic_exp_2_curve_4_[DATASET]_lam_[LAMBDA]_rand_[T]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L))
    L_comb = (d / t) * (((d - t) / (d - 1)) * L_diag + ((t - 1) / (d - 1)) * L)
    # Get stepsize matrix
    D =  np.linalg.inv(L_comb)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - D @ rand_t_sparsifier(grad_x, t)
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 2 curve 4", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_2_curve_4_" + dataset.split(sep='.')[0] + \
                  "_lam_{}_rand_{}_seed_{}.npy".format(lam, t, seed)
    outfile = os.path.join("result_exp_2", result_file)
    np.save(outfile, arrnm)
    return arrnm


# Functions for experiment 3, results can be found in folder './result_exp_3'
def run_cgd_curve_1_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with scalar stepsize, matrix smoothness, rand-1-uniform
    The average norm is solved as "logistic_exp_3_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix
    L_diag = np.diag(np.diag(L_matrix))
    max_eig_L_diag = np.linalg.eigh(L_diag)[0][-1]
    # Get stepsize matrix
    gamma_1 = (1 / d) * (1 / max_eig_L_diag)
    D = np.eye(d) * gamma_1
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_1 * special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 1", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_1_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_2_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running CGD with scalar stepsize, matrix smoothness, rand-1-importance
    The average norm is solved as "logistic_exp_3_curve_1_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """

    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix and probability
    L_tr    = np.trace(L_matrix)
    prob_p  = np.diag(L_matrix) / L_tr
    indices_imp_curve_2 = np.random.choice(d, size=iterations, replace=True, p=prob_p)
    # Get stepsize matrix
    gamma_2 = 1 / L_tr
    D = np.eye(d) * gamma_2
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - gamma_2 * special_rand_1(grad_x, idx=indices_imp_curve_2[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 2", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_2_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_3_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running algorithm 1 with diagonal matrix stepsize, matrix smoothness, rand-1-uniform/importance
    The average norm is solved as "logistic_exp_2_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix and probability
    L_diag = np.diag(np.diag(L_matrix))
    L_diag_inv = np.linalg.inv(L_diag)
    # Get stepsize matrix
    D = (1 / d) * L_diag_inv
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - D @ special_rand_1(grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 3", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_3_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_cgd_curve_5_exp_3(x_iter, L_matrix, A, b, iterations=100, lam=1, plot=False, save=True, dataset='null'):
    """
    Running algorithm 2 with diagonal matrix stepsize, matrix smoothness, rand-1-uniform/importance
    The average norm is solved as "logistic_exp_2_curve_3_[DATASET]_lam_[LAMBDA]_seed_[SEED].npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get shape
    n, d = A.shape
    # Calculate related matrix and probability
    L_diag = np.diag(np.diag(L_matrix))
    L_diag_inv = np.linalg.inv(L_diag)
    # Get stepsize matrix
    D = (1 / d) * L_diag_inv
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        # Update
        grad_x = cal_grad(A, b, x_iter, lam)
        x_iter = x_iter - special_rand_1(D @ grad_x, idx=indices[i])
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 3 curve 4", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_3_curve_5_" + dataset.split(sep='.')[0] + "_lam_{}_seed_{}.npy".format(lam, seed)
    outfile = os.path.join("result_exp_3", result_file)
    np.save(outfile, arrnm)
    return arrnm


# Functions for experiment 4, results can be found in folder './result_exp_4'
def find_min_GD(A, b, L, lam=0, iterations=200, plot=False, whole=False, index=-1):
    """
    Running GD to find the minimum of functions if we can not deduce its minimum
    Input:  iterations is the number of epochs of GD
    Output: the minimum encountered in the entire run of GD
    """
    # Shape
    n, d = A.shape
    # Iterate
    x_iter = np.ones(d)
    # Record
    fvals = []
    # Stepsize
    gamma = 1 / L
    min_f = 1000000
    for i in range(iterations):
        grad_x = cal_grad(A, b, x_iter, lam=lam)
        x_iter = x_iter - gamma * grad_x
        fvals.append(cal_func_val(A, b, x_iter, lam=lam))
        f_val = cal_func_val(A, b, x_iter, lam=lam)
        if min_f > f_val:
            min_f = f_val

    # Plotting to see if it is correct
    arrfv = np.array(fvals)
    if plot == True:
        iter_range = np.arange(1, arrfv.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Log Function values")
        plt.plot(iter_range, np.log10(arrfv), label="GD to find minimum", marker='o', markevery=500)
        # plt.legend()
        plt.show()
    # Return the minimum of the function
    # if whole == True:
    #     result_file = dataset.split(".")[0] + "_f_global_minimum_logistic_client_{}_" \
    #                   "iterations_{}_lam_{}_seed_{}.npy".format(num_client, iterations, lam, seed)
    # else:
    #     result_file = dataset.split(".")[0] + "_f_local_{}_minimum_logistic_client_{}_" \
    #                   "iterations_{}_lam_{}_seed_{}.npy".format(index, num_client, iterations, lam, seed)
    # outfile = os.path.join("Minimum", result_file)
    # np.save(outfile, arrfv)
    return min_f

def run_DCGD_curve_1_exp_4(x_iter, gamma, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    DCGD scalar stepsize, scalar smoothness
    The average norm is solved as "logistic_exp_4_curve_1_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get stepsize matrix
    D = gamma * np.eye(d)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - gamma * grad_est
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 4 curve 1", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_4_curve_1_" + dataset.split(sep='.')[0]\
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile = os.path.join("result_exp_4", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_DCGD_curve_2_exp_4(x_iter, gamma, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    DCGD scalar stepsize, matrix smoothness
    The average norm is solved as "logistic_exp_4_curve_2_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm = [], []
    # Get stepsize matrix
    D = gamma * np.eye(d)
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - gamma * grad_est
        # Recording
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
    # Plotting
    arrnm = np.array(avgnm)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="Experiment 4 curve 2", marker='o', markevery=250)
        plt.legend()
        plt.show()
    # Saving
    result_file = "logistic_exp_4_curve_2_" + dataset.split(sep='.')[0]\
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile = os.path.join("result_exp_4", result_file)
    np.save(outfile, arrnm)
    return arrnm

def run_DCGD_curve_3_exp_4(x_iter, D, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    Running algorithm 1 with optimal diagonal matrix stepsize, matrix smoothness
    Matrix norm result:
    The average norm is solved as "logistic_exp_4_curve_3_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    Standard Euclidean norm result:
    The average norm is solved as "std_logistic_exp_4_curve_3_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []
    # Get stepsize matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - D @ grad_est
        # Recording matrix norm
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
        # Recording euclidean norm
        stdm_iter = np.linalg.norm(grad_x) ** 2
        stdnm.append(stdm_iter)
        avgstd.append(np.average(stdnm))
    # Plotting
    arrnm = np.array(avgnm)
    arrstd = np.array(avgstd)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="s1", marker='o', markevery=2000)
        plt.plot(iter_range, arrstd, label="s2", marker='o', markevery=2000)
        plt.legend()
        plt.show()
    # Saving
    result_file_1 = "logistic_exp_4_curve_3_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    result_file_2 = "std_logistic_exp_4_curve_3_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile_1 = os.path.join("result_exp_4", result_file_1)
    outfile_2 = os.path.join("result_exp_4", result_file_2)
    np.save(outfile_1, arrnm)
    np.save(outfile_2, arrstd)
    return arrnm, arrstd

def run_DCGD_curve_4_exp_4(x_iter, D, num_client, iterations=100, lam=0, plot=False, save=True, dataset="null"):
    """
    Running algorithm 2 with optimal diagonal matrix stepsize, matrix smoothness
    Matrix norm result:
    The average norm is solved as "logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    Standard Euclidean norm result:
    The average norm is solved as "std_logistic_exp_4_curve_4_[DATASET]_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy"
    """
    # Recording norms per iteration
    norms, avgnm, stdnm, avgstd = [], [], [], []
    # Get stepsize matrix
    D_1d = sp.linalg.fractional_matrix_power(D, 1 / d)
    D_1d_det = sp.linalg.det(D_1d)
    # Iterations
    for i in range(iterations):
        grad_est = np.zeros_like(x_iter)
        for c in range(num_client):
            # Gradient for each client
            grad_x_c = cal_grad(split_A[c], split_b[c], x_iter, lam)
            # Compressed
            comp_grad_c = rand_t_sparsifier(D @ grad_x_c, t=1)
            grad_est += comp_grad_c
        # Get average
        grad_est = grad_est / num_client
        # Record the gradient at this iteration
        grad_x = cal_grad(A, b, x_iter, lam)
        # Update the iterate
        x_iter = x_iter - grad_est
        # Recording matrix norm
        norm_iter = cal_matrix_norm(D, grad_x) / D_1d_det
        norms.append(norm_iter)
        avgnm.append(np.average(norms))
        # Recording euclidean norm
        stdm_iter = np.linalg.norm(grad_x) ** 2
        stdnm.append(stdm_iter)
        avgstd.append(np.average(stdnm))
    # Plotting
    arrnm = np.array(avgnm)
    arrstd = np.array(avgstd)
    if plot == True:
        iter_range = np.arange(1, arrnm.shape[0] + 1)
        plt.xlabel("Iterations")
        plt.ylabel("Lognorm")
        plt.yscale("log")
        plt.plot(iter_range, arrnm, label="s1", marker='o', markevery=2000)
        plt.plot(iter_range, arrstd, label="s2", marker='o', markevery=2000)
        plt.legend()
        plt.show()
    # Saving
    result_file_1 = "logistic_exp_4_curve_4_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    result_file_2 = "std_logistic_exp_4_curve_4_" + dataset.split(sep='.')[0] \
                  + "_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(lam, num_client, seed, iterations, epsilon_sq)
    outfile_1 = os.path.join("result_exp_4", result_file_1)
    outfile_2 = os.path.join("result_exp_4", result_file_2)
    np.save(outfile_1, arrnm)
    np.save(outfile_2, arrstd)
    return arrnm, arrstd


# Parameters
parser = argparse.ArgumentParser(description='Experiment')
parser.add_argument('--exp_index', '-i', help="Index of the experiment", default=1, required=True)
parser.add_argument('--dataset', '-d', help='Name of dataset', default='Null', required=True)
parser.add_argument('--lambda_val', '-l', help='Value of lambda', default=0.2)
parser.add_argument('--client', '-c', help='Number of client', default=1)
parser.add_argument('--seed', '-s', default=522)
parser.add_argument('--eps', '-e', help='Error level', default=0.0001)
parser.add_argument('--iterations', '-K', help='Number of iterations to run', default=3000)
args = parser.parse_args()


if __name__ == "__main__":

    # Hyperparameters
    cur_exp = int(args.exp_index)
    iterations = int(args.iterations)
    lam = float(args.lambda_val)
    num_client = int(args.client)
    epsilon_sq = float(args.eps)
    dataset = args.dataset + '.txt'
    PLOT = False

    # Test
    # print(cur_exp, type(cur_exp))
    # print(dataset, type(dataset))
    # print(lam, type(lam))
    # print(iterations, type(iterations))
    # exit()


    # Load the a1a dataset, for a1a, A: (1605, 119); b: (1605, )
    path_dataset = os.path.join('dataset', dataset)
    A_train_a1a, b_train_a1a = load_svmlight_file(path_dataset)

    # Turning into numpy arrays
    A, b = A_train_a1a.toarray(), b_train_a1a
    n, d = A.shape

    # Calculating L smoothness matrix
    sum_mat, lam_min, lam_max = cal_sum_rank_1(A)[0], cal_sum_rank_1(A)[1], cal_sum_rank_1(A)[2]

    # Constructing L using different lambda
    L = sum_mat + 2 * lam * np.eye(d)

    # Show eigen value histogram
    if PLOT == True:
        # plot_eigs(L)
        pass

    # Calculating related matrices and scalars
    L_diag          = np.diag(np.diag(L))
    L_diag_inv      = sp.linalg.inv(L_diag)
    L_inv           = sp.linalg.inv(L)
    L_inv_det       = sp.linalg.det(L_inv)
    L_half          = sp.linalg.fractional_matrix_power(L, (1 / 2))
    L_half_inv      = sp.linalg.inv(L_half)

    max_eig_L       = np.linalg.eigh(L)[0][-1]
    max_eig_L_diag  = np.linalg.eigh(L_diag)[0][-1]

    # Choosing random seed and initial point
    seed = 522
    np.random.seed(seed)
    x_initial = np.ones(d) / np.sqrt(d)

    # Rand-1 sequence
    indices = np.random.randint(low=0, high=d, size=iterations)

    # Experiment 1
    if cur_exp == 1:
        ############################    Experiments 1: rand-1 case first four
        # Curve 1: Running CGD with scalar stepsize, scalar smoothness, rand-1
        arrnm_1 = run_cgd_curve_1_exp_1(x_initial, max_eig_L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 2: Running CGD with scalar stepsize, matrix smoothness, rand-1
        arrnm_2 = run_cgd_curve_2_exp_1(x_initial, max_eig_L_diag, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 3: Running CGD with diagonal matrix stepsize, matrix smoothness, rand-1, algorithm 1
        arrnm_3 = run_cgd_curve_3_exp_1(x_initial, L_diag_inv, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 4: Running CGD with diagonal matrix stepsize, matrix smoothness, rand-1, algorithm 2
        arrnm_4 = run_cgd_curve_4_exp_1(x_initial, L_diag_inv, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)

        ############################    Experiment 1: GD benchmark
        # Curve 5: Running GD with scalar stepsize, scalar smoothness
        arrnm_5 = run_gd_scalar(x_initial, max_eig_L, A, b, iterations, lam,
                                plot=False, save=True, dataset=dataset)
        # Curve 6: Running GD with matrix stepsize, matrix smoothness
        arrnm_6 = run_gd_matrix(x_initial, L_inv, A, b, iterations, lam,
                                plot=False, save=True, dataset=dataset)

        ############################    Experiments 1: rand-1 case last 2
        # Curve 7: Running CGD with L-1 matrix stepsize, matrix smoothness, rand-1, algorithm 1
        arrnm_7 = run_cgd_curve_7_exp_1(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 8: Running CGD with L-1/2 matrix stepsize, matrix smoothness, rand-1, algorithm 1
        arrnm_8 = run_cgd_curve_8_exp_1(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)

        if PLOT == True:
            iterations = np.arange(1, arrnm_3.shape[0] + 1)
            plt.xlabel("Iterations")
            plt.ylabel("Log average norm")
            plt.plot(iterations, np.log10(arrnm_1),
                     label="CGD, scalar stepsize, scalar smoothness, rand-1, lam={}".format(lam),
                     marker='o', markevery=250)
            plt.plot(iterations, np.log10(arrnm_2),
                     label="CGD, scalar stepsize, matrix smoothness, rand-1, lam={}".format(lam),
                     marker='v', markevery=250)
            plt.plot(iterations, np.log10(arrnm_3),
                     label="Alg1, diagonal matrix stepsize, matrix smoothness, rand-1, lam={}".format(lam),
                     marker='^', markevery=250)
            plt.plot(iterations, np.log10(arrnm_4),
                     label="Alg2, diagonal matrix stepsize, matrix smoothness, rand-1, lam={}".format(lam),
                     marker='.', markevery=250)
            # plt.plot(iterations, np.log10(arrnm_5),
            #          label="GD, scalar stepsize, scalar smoothness, lam={}".format(lam),
            #          marker=',', markevery=250)
            # plt.plot(iterations, np.log10(arrnm_6),
            #          label="GD, matrix stepsize, matrix smoothness, lam={}".format(lam),
            #          marker='<', markevery=250)
            plt.plot(iterations, np.log10(arrnm_7),
                     label="Alg1, L-1 based stepsize, matrix smoothness, rand-1, lam={}".format(lam),
                     marker='>', markevery=250)
            plt.plot(iterations, np.log10(arrnm_8),
                     label="Alg1, L-1/2 based stepsize, matrix smoothness, rand-1, lam={}".format(lam),
                     marker='1', markevery=250)
            plt.legend()
            plt.savefig(os.path.join("Temps" ,"Exp_1_" + dataset.split(sep='.')[0] + "_lam_{}".format(lam) + ".png"))
            plt.show()

    # Experiment 2
    if cur_exp == 2:
        # control the minibatch
        t1, t2, t3 = int(d / 4) + 1, int(d / 2) + 1, int(3 * d / 4) + 1     # Several t we want to examine
        ############################    Experiments 2: rand-t1 case
        # Curve 1: Running CGD with scalar stepsize, scalar smoothness, rand-t1
        arrnm_1 = run_cgd_curve_1_exp_2(x_initial, max_eig_L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)
        # Curve 2: Running CGD with scalar stepsize, matrix smoothness, rand-t1
        arrnm_2 = run_cgd_curve_2_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)
        # Curve 3: Running alg2 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_3 = run_cgd_curve_3_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)
        # Curve 3: Running alg1 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_4 = run_cgd_curve_4_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t1)

        # Curve 1: Running CGD with scalar stepsize, scalar smoothness, rand-t1
        arrnm_5 = run_cgd_curve_1_exp_2(x_initial, max_eig_L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)
        # Curve 2: Running CGD with scalar stepsize, matrix smoothness, rand-t1
        arrnm_6 = run_cgd_curve_2_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)
        # Curve 3: Running alg2 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_7 = run_cgd_curve_3_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)
        # Curve 3: Running alg1 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_8 = run_cgd_curve_4_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t2)

        # Curve 1: Running CGD with scalar stepsize, scalar smoothness, rand-t1
        arrnm_9 = run_cgd_curve_1_exp_2(x_initial, max_eig_L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)
        # Curve 2: Running CGD with scalar stepsize, matrix smoothness, rand-t1
        arrnm_10 = run_cgd_curve_2_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)
        # Curve 3: Running alg2 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_11 = run_cgd_curve_3_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)
        # Curve 3: Running alg1 with optimal matrix stepsize, matrix smoothness, rand-t1
        arrnm_12 = run_cgd_curve_4_exp_2(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset, t=t3)



        if PLOT == True:
            iterations = np.arange(1, arrnm_1.shape[0] + 1)
            plt.xlabel("Iterations")
            plt.ylabel("Log average norm")
            plt.plot(iterations, np.log10(arrnm_1),
                     label="CGD, scalar stepsize, scalar smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='o', markevery=250)
            plt.plot(iterations, np.log10(arrnm_2),
                     label="CGD, scalar stepsize, matrix smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='v', markevery=250)
            plt.plot(iterations, np.log10(arrnm_3),
                     label="Alg2, optimal matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='<', markevery=250)
            plt.plot(iterations, np.log10(arrnm_4),
                     label="Alg1, same matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t1, lam),
                     marker='^', markevery=250)
            plt.plot(iterations, np.log10(arrnm_5),
                     label="CGD, scalar stepsize, scalar smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='1', markevery=250)
            plt.plot(iterations, np.log10(arrnm_6),
                     label="CGD, scalar stepsize, matrix smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='2', markevery=250)
            plt.plot(iterations, np.log10(arrnm_7),
                     label="Alg2, optimal matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='3', markevery=250)
            plt.plot(iterations, np.log10(arrnm_8),
                     label="Alg1, same matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t2, lam),
                     marker='4', markevery=250)
            plt.plot(iterations, np.log10(arrnm_9),
                     label="CGD, scalar stepsize, scalar smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='H', markevery=250)
            plt.plot(iterations, np.log10(arrnm_10),
                     label="CGD, scalar stepsize, matrix smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='+', markevery=250)
            plt.plot(iterations, np.log10(arrnm_11),
                     label="Alg2, optimal matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='_', markevery=250)
            plt.plot(iterations, np.log10(arrnm_12),
                     label="Alg1, same matrix stepsize, matrix smoothness, rand-{}, lam={}".format(t3, lam),
                     marker='8', markevery=250)
            plt.legend()
            plt.savefig(os.path.join("Temps" ,"Exp_2_" + dataset.split(sep='.')[0] +
                                     "_lam_{}_rand_{}".format(lam, t1) + ".png"))
            plt.show()

    # Experiment 3
    if cur_exp == 3:
        # Curve 1: uniform with scalar stepsize
        arrnm_1 = run_cgd_curve_1_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 2: uniform with scalar stepsize
        arrnm_2 = run_cgd_curve_2_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 3: algorithm 1 with uniform/importance with diagonal matrix stepsize
        # Note that the importance sampling probability is uniform in this case
        arrnm_3 = run_cgd_curve_3_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)
        # Curve 5: algorithm 2 with uniform/importance with diagonal matrix stepsize
        # Note that the importance sampling probability is uniform in this case
        arrnm_5 = run_cgd_curve_5_exp_3(x_initial, L, A, b, iterations, lam,
                                        plot=False, save=True, dataset=dataset)


        if PLOT == True:
            iterations = np.arange(1, arrnm_1.shape[0] + 1)
            plt.xlabel("Iterations")
            plt.ylabel("Log average norm")
            plt.plot(iterations, np.log10(arrnm_1),
                     label="CGD, scalar stepsize, matrix smoothness, rand-1-uniform, lam={}".format(lam),
                     marker='o', markevery=250)
            plt.plot(iterations, np.log10(arrnm_2),
                     label="CGD, scalar stepsize, matrix smoothness, rand-1-importance, lam={}".format(lam),
                     marker='1', markevery=250)
            plt.plot(iterations, np.log10(arrnm_3),
                     label="alg1, diagonal matrix stepsize, matrix smoothness, rand-1-imp/uniform, lam={}".format(lam),
                     marker='<', markevery=250)
            plt.plot(iterations, np.log10(arrnm_5),
                     label="alg2, diagonal matrix stepsize, matrix smoothness, rand-1-imp/uniform, lam={}".format(lam),
                     marker='v', markevery=200)
            plt.legend()
            plt.savefig(os.path.join("Temps" ,"Exp_3_" + dataset.split(sep='.')[0] +
                                     "_lam_{}_rand_1".format(lam) + ".png"))
            plt.show()

    # Experiment 4
    if cur_exp == 4:
        # Reshuffling the dataset
        # A, b = shuffle(A, b, random_state=0)
        # Splitting the dataset into smaller part
        split_A, split_b = np.array_split(A, num_client), np.array_split(b, num_client)

        # Calculating the smoothness matrix for function g alone
        L_g       = cal_sum_rank_1(A)[0]
        split_L_g = [cal_sum_rank_1(split_A[i])[0]  for i in range(num_client)]
        # Calculating the smoothness matrix for the whole function
        L         = L_g + 2 * lam * np.eye(d)
        split_L   = [split_L_g[i] + 2 * lam * np.eye(d) for i in range(num_client)]

        # Finding the maximum eigenvalue of the g part
        L_g_scalar = np.linalg.eigh(L_g)[0][-1]
        split_L_g_scalar = [np.linalg.eigh(split_L_g[i])[0][-1] for i in range(num_client)]

        # For convenience, we set lam = 0, now it becomes a convex objective to find the minimum
        # Finding minimum
        min_f = find_min_GD(A, b, L_g_scalar, lam=lam, plot=False, whole=True)
        split_min_f = [find_min_GD(split_A[i], split_b[i], split_L_g_scalar[i],
                                   lam=lam, plot=False, whole=False, index=i) for i in range(num_client)]

        # Setting the relative error level
        # epsilon_sq = 0.0001
        # Some values needed
        # Because we are using rand-1 sketch
        omega           = d - 1
        # Remember we have a regularizer in this case
        L_scalar        = L_g_scalar + 2 * lam
        # lam does not affect which one is the biggest
        L_max_scalar    = np.max(split_L_g_scalar) + 2 * lam
        # C should be larger than 0 （This is an upper bound on C）
        C = (2 / num_client) * (min_f - np.average(split_min_f))
        # assert C >= 0, "Assertion error, negative C={}".format(C)


        # For curve 1:
        upper_bound = [
            (1 / L_scalar),
            np.sqrt(num_client) / np.sqrt(omega * L_scalar * L_max_scalar * iterations),
            epsilon_sq / (2 * L_scalar * L_max_scalar * omega * C)
        ]
        # This is the largest gamma we can demand
        gamma = np.min(upper_bound)
        arrnm_1 = run_DCGD_curve_1_exp_4(x_initial, gamma, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)


        # For curve 2:
        # Upper bound 1
        upper_bound_1   = (1 / L_scalar)
        # Upper bound 2
        diff_L          = d * np.diag(np.diag(L)) - L
        split_L_half    = [sp.linalg.fractional_matrix_power(split_L[i], (1/2)) for i in range(num_client)]
        eigen_list      = [np.linalg.eigh(split_L_half[i] @ diff_L @ split_L_half[i])[0][-1] for i in range(num_client)]
        cond_2_max      = np.max(eigen_list)
        upper_bound_2   = np.sqrt(n) / np.sqrt(iterations * cond_2_max)
        # Upper bound 3
        cond_3_max     = cond_2_max
        upper_bound_3   = epsilon_sq / (2 * C * cond_3_max)
        # Merging upperbound
        upper_bound = [upper_bound_1, upper_bound_2, upper_bound_3]
        gamma = np.min(upper_bound)
        arrnm_2 = run_DCGD_curve_2_exp_4(x_initial, gamma, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)


        # For curve 3:
        L_diag          = np.diag(np.diag(L))
        L_diag_inv      = sp.linalg.inv(L_diag)
        L_diag_inv_half = sp.linalg.fractional_matrix_power(L_diag, -(1/2))
        # Upper bound 1
        upper_bound_1   = (1 / np.linalg.eigh(L_diag_inv_half @ L @ L_diag_inv_half)[0][-1])
        # Upper bound 2
        diff_L_alg_1 = d * L_diag_inv - L_diag_inv @ L @ L_diag_inv
        matrix_list_alg_1 = [split_L_half[i] @ diff_L_alg_1 @ split_L_half[i] for i in range(num_client)]
        max_list_eig_1 = np.max([np.linalg.eigh(matrix_list_alg_1[i])[0][-1] for i in range(num_client)])
        upper_bound_2 = np.sqrt(n) / np.sqrt(iterations * max_list_eig_1)
        # Upper bound 3
        L_diag_inv_1d = sp.linalg.fractional_matrix_power(L_diag_inv, 1/d)
        det_rhs       = sp.linalg.det(L_diag_inv_1d)
        upper_bound_3 = epsilon_sq / (2 * C * det_rhs)
        # Merging upperbound
        upper_bound = [upper_bound_1, upper_bound_2, upper_bound_3]
        gamma = np.min(upper_bound)
        # Getting stepsize
        D = gamma * L_diag_inv
        arrnm_3, arrstd_3 = run_DCGD_curve_3_exp_4(x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)


        # For curve 4: it is the same with curve 3 in this case
        arrnm_4, arrstd_4 = run_DCGD_curve_4_exp_4(x_initial, D, num_client, iterations,
                                         lam=lam, plot=False, save=True, dataset=dataset)
        if PLOT == True:
            raise NotImplementedError
            iterations = np.arange(1, arrnm_1.shape[0] + 1)
            # plt.xlabel("Iterations")
            # plt.ylabel("Log average norm")
            # plt.plot(iterations, np.log10(arrnm_1),
            #          label="Standard DCGD".format(lam),
            #          marker='o', markevery=1000)
            # plt.plot(iterations, np.log10(arrnm_2),
            #          label="DCGD matrix smoothness".format(lam),
            #          marker='1', markevery=1000)
            # plt.plot(iterations, np.log10(arrnm_3),
            #          label="Algorithm 1".format(lam),
            #          marker='<', markevery=1000)
            # plt.plot(iterations, np.log10(arrnm_4),
            #          label="Algorithm 2".format(lam),
            #          marker='>', markevery=1000)
            # plt.plot(iterations, np.log10(arrstd_3),
            #          label="Algorithm 1 Euc".format(lam),
            #          marker='<', markevery=1100)
            # plt.plot(iterations, np.log10(arrstd_4),
            #          label="Algorithm 2 Euc".format(lam),
            #          marker='>', markevery=900)
            # plt.legend()
            # plt.savefig(os.path.join("Temps" ,"Exp_4_" + dataset.split(sep='.')[0] +
            #                          "_lam_{}_num_client_{}_rand_1".format(lam, num_client) + ".png"))
            # plt.show()













