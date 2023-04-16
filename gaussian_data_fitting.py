from utils import part1_preprocess, poly_basis_func
from plot_utils import plot_poly_degree, plot_training_batch, plot_training_iter, plot_training_lambda, training_loss_func
import numpy as np


if __name__ == '__main__':

    X, t = part1_preprocess('gaussian.csv')
    plot_poly_degree(X, t, 'Plot_Degree_100_G.png', method='pinv')
    Phi_X = poly_basis_func(X, 7)
    idx = np.arange(0, 21, 1)
    plot_poly_degree(X[idx], t[idx], 'Plot_Degree_20_G.png', method='pinv')

    rng = np.random.default_rng(3591)
    idx_perm = rng.permutation(idx)
    train_idx = idx_perm[:15]
    test_idx = idx[15:]

    idx100 = rng.permutation(100)

    plot_training_lambda(Phi_X, t, 'Lambda_Plot_100_G.png')
    plot_training_lambda(Phi_X[idx], t[idx], 'Lambda_Plot_20_G.png')
    plot_training_iter(Phi_X[train_idx], t[train_idx], Phi_X[test_idx], t[test_idx], lamb=1e-4, lr=1e-4, fname='Training_Epoch_Plot_20_GD_G.png')
    plot_training_iter(Phi_X[idx100[:70]], t[idx100[:70]], Phi_X[idx100[70:]], t[idx100[70:]], lamb=1e-4, lr=1e-4, fname='Training_Epoch_Plot_100_GD_G.png')
    plot_training_batch(Phi_X[idx], t[idx], lamb=1e-4, fname='Batch_Size_Plot_20_G.png')
    plot_training_batch(Phi_X, t, lamb=1e-4, fname='Batch_Size_plot_100_G.png')
    training_loss_func(Phi_X[idx], t[idx], 1e-4, 'Results_Various_Loss_Func_20_G.csv')
    training_loss_func(Phi_X, t, 1e-4, 'Results_Various_Loss_Func_100_G.csv')
    