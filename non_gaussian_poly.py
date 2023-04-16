from LinReg import LinearRegression
from utils import poly_basis_func, part1_preprocess
from plot_utils import calc_noise, plot_func, training_cross_val

X, t = part1_preprocess('non_gaussian.csv')
Phi_X = poly_basis_func(X, 9)

print(training_cross_val(Phi_X, t, bs=None, lamb=1e-2, method='pinv'))

LR = LinearRegression(method='pinv', batch_size=30, p=2, q=2, learning_rate=1e-3, rms=0.999, beta=0.9, lambd=1e-2)
LR.fit(Phi_X, t, verbosity=5000, max_iter=50000, tol=0)

with open('Weights_Non_Gaussian_100.csv', 'w+') as f:
    for wt in LR.w:
        f.write(str(wt)+'\n')

plot_func(X, t, lambda x: poly_basis_func(x, 9), LR, 'Non_Gaussian_Polynomial_Pinv.png')
calc_noise(Phi_X, t, LR, 'Noise-Distribution_NG.png')