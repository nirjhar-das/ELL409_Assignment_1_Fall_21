from LinReg import LinearRegression
from utils import poly_basis_func, part1_preprocess
from plot_utils import plot_func, calc_noise, training_cross_val

X, t = part1_preprocess('gaussian.csv')
Phi_X = poly_basis_func(X, 7)
Phi_X_20 = Phi_X[:20, :]
t_20 = t[:20]

#print(training_cross_val(Phi_X, t, lamb=0, bs=None, method='pinv'))
#print(training_cross_val(Phi_X_20, t_20, lamb=0, bs=None, method='pinv'))
#print(training_cross_val(Phi_X, t, lamb=1e-4, bs=None, method='pinv'))
#print(training_cross_val(Phi_X_20, t_20, lamb=1e-4, bs=None, method='pinv'))

LR1 = LinearRegression(method='pinv', batch_size=100, p=2, q=2, learning_rate=1e-2, rms=0.999, beta=0.9, lambd=1e-4)
LR1.fit(Phi_X, t, verbosity=500, max_iter=50000, tol=0)

LR2 = LinearRegression(method='pinv', batch_size=100, p=2, q=2, learning_rate=1e-2, rms=0.999, beta=0.9, lambd=1e-4)
LR2.fit(Phi_X_20, t_20, verbosity=500, max_iter=50000, tol=0)

with open('Weights_Gaussian_100.csv', 'w+') as f:
    for wt in LR1.w:
        f.write(str(wt)+'\n')

with open('Weights_Gaussian_20.csv', 'w+') as f:
    for wt in LR2.w:
        f.write(str(wt)+'\n')


plot_func(X[:20], t_20, lambda x: poly_basis_func(x, 7), LR2, 'Gaussian_Polynomial_Pinv_20.png')
plot_func(X, t, lambda x: poly_basis_func(x, 7), LR1, 'Gaussian_Polynomial_Pinv_100.png')
calc_noise(Phi_X_20, t_20, LR2, 'Noise-Distribution_G_20.png')
calc_noise(Phi_X, t, LR1, 'Noise-Distribution_G_100.png')