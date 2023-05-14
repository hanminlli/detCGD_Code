import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['text.usetex'] = True
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('text.latex', preamble=r'\usepackage{cmbright}')


parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('--exp_index', '-i', help="Index of the experiment", default=1, required=True)
parser.add_argument('--dataset', '-d', help='Name of dataset', default='Null', required=True)
parser.add_argument('--lambda_val', '-l', help='Value of lambda', default=0.2)
parser.add_argument('--client', '-c', help='Number of client', default=1)
parser.add_argument('--seed', '-s', default=522)
parser.add_argument('--eps', '-e', help='Error level', default=0.0001)
parser.add_argument('--iterations', '-K', help='Number of iterations to run', default=3000)
parser.add_argument('--minibatch', '-t', help='Minibatch size', default=1)
args = parser.parse_args()

if __name__ == "__main__":
    # Setting for experiment 4
    plot_Euc = False    # Set to true if we want to plot the result in terms of standard euclidean norm for experiment 4

    # Hyperparameters
    exp_index   = int(args.exp_index)
    dataset     = args.dataset
    lam         = float(args.lambda_val)
    client      = int(args.client)
    seed        = int(args.seed)
    eps         = float(args.eps)
    iterations  = int(args.iterations)
    t           = int(args.minibatch)
    font1 = {'family': 'New Times Romans', 'weight': 'normal', 'size': 18, }
    font2 = {'family': 'New Times Romans', 'weight': 'normal', 'size': 12, }
    font3 = {'family': 'New Times Romans', 'weight': 'normal', 'size': 14,}

    if exp_index == 1:
        result_dir = "result_exp_1"
        filename = ["logistic_exp_1_curve_{}_{}_lam_{}_seed_{}.npy".format(
            i, dataset, lam, seed
        ) for i in range(1, 9)]
        arrnm = [np.load(os.path.join(result_dir, filename[i])) for i in range(len(filename))]
        iterations = np.arange(1, arrnm[0].shape[0] + 1)

        # Plotting
        plt.figure(figsize=(6.5, 5.5))
        plt.xticks(fontname="New Times Romans")
        plt.yticks(fontname="New Times Romans")
        plt.xlabel(r"Iterations", font1)
        plt.ylabel(r"$G_{K, \bf{D}}$", font1)
        plt.yscale('log')
        plt.title(r"{}, rand-$1$ sketch, $\lambda={}$".format(dataset, lam), font1)
        plt.plot(iterations, arrnm[0],
                 label=r"Standard CGD",
                 marker='o', markevery=500, linestyle=(0, (1, 1)))
        plt.plot(iterations, arrnm[1],
                 label=r"CGD-mat",
                 marker='v', markevery=500, linestyle='dashed')
        plt.plot(iterations, arrnm[2],
                 label=r"det-CGD$1$ with $\bf{D}_1$",
                 marker='<', markevery=600, linestyle='dotted')
        plt.plot(iterations, arrnm[3],
                 label=r"det-CGD$2$ with $\bf{D}_2$",
                 marker='^', markevery=400, linestyle='dashdot')
        # plt.plot(iterations, arrnm[4],
        #          label=r"Standard GD",
        #          marker='>', markevery=500, linestyle=(0, (1, 1)))
        # plt.plot(iterations, arrnm[5],
        #          label=r"GD with matrix smoothness",
        #          marker='1', markevery=250, linestyle='dashed')
        plt.plot(iterations, arrnm[6],
                 label=r"det-CGD$1$ with $\bf{D}_3$",
                 marker='2', markevery=500, linestyle=(0, (5, 1)))
        plt.plot(iterations, arrnm[7],
                 label=r"det-CGD$1$ with $\bf{D}_4$",
                 marker='>', markevery=550, linestyle=(0, (3, 1, 1, 1)))
        plt.grid(axis='x', linestyle='dashed')
        plt.legend(prop=font3)
        # plt.savefig("Exp_1_lam_{}".format(lam) + dataset + ".pdf")
        plt.savefig(os.path.join('figures', "Exp_1_lam_{}_".format(lam) + dataset + ".png"))
        # plt.show()

    if exp_index == 2:
        result_dir = "result_exp_2"
        filename = ["logistic_exp_2_curve_{}_{}_lam_{}_rand_{}_seed_522.npy".format(
            i, dataset, lam, t
        ) for i in range(1, 5)]
        arrnm = [np.load(os.path.join(result_dir, filename[i])) for i in range(len(filename))]
        iterations = np.arange(1, arrnm[0].shape[0] + 1)

        # Plotting
        plt.figure(figsize=(6.5, 5.5))
        plt.xticks(fontname="New Times Romans")
        plt.yticks(fontname="New Times Romans")
        plt.xlabel(r"Iterations", font1)
        plt.ylabel(r"$G_{K, \bf{D}}$", font1)
        plt.yscale('log')
        plt.title(r"{}, rand-${}$ sketch, $\lambda={}$".format(dataset, t ,lam), font1)
        plt.plot(iterations, arrnm[0],
                 label=r"Standard CGD",
                 marker='o', markevery=500, linestyle=(0, (1, 1)))
        plt.plot(iterations, arrnm[1],
                 label=r"CGD-mat",
                 marker='v', markevery=500, linestyle='dashed')
        plt.plot(iterations, arrnm[2],
                 label=r"det-CGD$1$ with $\bf{D}$",
                 marker='<', markevery=600, linestyle='dotted')
        plt.plot(iterations, arrnm[3],
                 label=r"det-CGD$2$ with $\bf{D}$",
                 marker='^', markevery=400, linestyle='dashdot')
        plt.legend(prop=font3)
        # plt.savefig("Exp_2_lam_{}".format(lam) + dataset + ".pdf")
        plt.savefig(os.path.join('figures', "Exp_2_lam_{}_rand_{}_".format(lam, t) + dataset + ".png"))
        # plt.show()

    if exp_index == 3:
        raise NotImplementedError

    if exp_index == 4:
        result_dir = "result_exp_4"
        # For ploting in the standard Euclidean norm, just need to add "std_" for the last two file as a prefix
        filename_mat = ["logistic_exp_4_curve_{}_{}_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(
            i, dataset, lam, client, seed, iterations, eps
        ) for i in range(1, 5)]

        if plot_Euc == True:
            filename_mat = ["logistic_exp_4_curve_{}_{}_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(
            i, dataset, lam, client, seed, iterations, eps
            ) for i in range(1, 3)] + \
                           ["std_logistic_exp_4_curve_{}_{}_lam_{}_client_{}_seed_{}_iter_{}_epi_{}.npy".format(
                i, dataset, lam, client, seed, iterations, eps
            ) for i in range(3, 5)]

        arrnm = [np.load(os.path.join(result_dir, filename_mat[i])) for i in range(len(filename_mat))]
        iterations = np.arange(1, arrnm[0].shape[0] + 1)

        plt.figure(figsize=(6.5, 5.5))
        plt.xticks(fontname="New Times Romans")
        plt.yticks(fontname="New Times Romans")
        plt.xlabel(r"Iterations", font1)
        if plot_Euc == False:
            plt.ylabel(r"$G_{K, \bf{D}}$", font1)
        else:
            plt.ylabel(r"$E_{K}$", font1)
        plt.yscale('log')
        plt.title(r"{}, rand-$1$ sketch, $\lambda={}$, $n={}$".format(dataset, lam, client), font1)
        plt.plot(iterations, arrnm[0],
                 label=r"Standard DCGD",
                 marker='o', markevery=300, linestyle=(0, (1, 1)))
        plt.plot(iterations, arrnm[1],
                 label=r"DCGD-mat",
                 marker='v', markevery=300, linestyle='dashed')
        plt.plot(iterations, arrnm[2],
                 label=r"D det-CGD$1$ diagonal $\bf{D}$",
                 marker='^', markevery=400, linestyle='dashdot')
        plt.plot(iterations, arrnm[3],
                 label=r"D det-CGD$2$ diagonal $\bf{D}$",
                 marker='<', markevery=400, linestyle='dotted')
        plt.grid(axis='x', linestyle='dashed')
        plt.legend(prop=font3)
        plt.savefig(os.path.join('figures', "Exp_4_lam_{}_client_{}_".format(lam, client) + dataset + ".png"))
        # plt.show()



