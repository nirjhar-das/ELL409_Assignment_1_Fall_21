from utils import part1_preprocess, poly_basis_func
from plot_utils import plot_poly_degree, plot_training_batch, plot_training_iter, plot_training_lambda, training_loss_func
import numpy as np


if __name__ == '__main__':

    X, t = part1_preprocess('non_gaussian.csv')
    plot_poly_degree(X, t, 'Plot_Degree_100_NG.png')
    Phi_X = poly_basis_func(X, 9)
    
    rng = np.random.default_rng(3591)

    idx100 = rng.permutation(100)

    training_loss_func(Phi_X, t, 1e-5, 'Results_Various_Loss_Func_100_NG.csv')
    plot_training_lambda(Phi_X, t, 'Lambda_Plot_100_NG.png')
    plot_training_iter(Phi_X[idx100[:70]], t[idx100[:70]], Phi_X[idx100[70:]], t[idx100[70:]], lamb=1e-5, lr=1e-3, fname='Training_Epoch_Plot_100_GD_NG.png')
    plot_training_batch(Phi_X, t, lamb=1e-5, fname='Batch_Size_plot_100_NG.png')
    