from plot_utils import plot_func, plot_training_iter, training_cross_val
from LinReg import LinearRegression
import pandas as pd
import numpy as np
from utils import fourier_basis_func, poly_basis_func, wave_basis_func

def preprocess(df):
    df['date'] = pd.to_datetime(df['id'])
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(inplace=True)
    base = df['date'][0]
    df['n'] = df['date'].apply(lambda x: (x.to_period('M') - base.to_period('M')).n)

def postprocess(y_hat, date_list, fname='test.csv', output_fname='Result.csv'):
    res_dict = {date_list[i] : y_hat[i] for i in range(len(date_list))}
    df_res = pd.read_csv(fname)
    df_res['value'] = [res_dict[k] for k in df_res['id']]
    df_res.to_csv(output_fname, index=False)

if __name__=='__main__':

    df = pd.read_csv('train.csv')
    preprocess(df)

    X = np.array(df['n'])
    t = np.array(df['value'])

    #X_year = X.astype(int)//12

    #Phi_Xyr_poly = poly_basis_func(X_year, 3)[:, 1:]

    #Phi_X = wave_basis_func(X, 50, 120)
    #Phi_X_new = np.concatenate([Phi_X, Phi_Xyr_poly], axis=1)
    Phi_X_F = fourier_basis_func(X, 20, 120)
    #mu = t.mean()
    #t = t - t.mean()
    #tr_rmse, te_rmse, _ = training_cross_val(Phi_X_F, t, bs=1, lamb=1e-2, lr=1e-3, method='pinv')
    #print(tr_rmse, te_rmse)

    #LR = LinearRegression(method='pinv', batch_size=1, p=2, q=2, learning_rate=1e-2, lambd=1e-2, decay=0.001, beta=0.8)

    #LR.fit(Phi_X_F, t, verbosity=250, tol=1e-8)

    #print(LR.loss, LR.rmse)

    rng = np.random.default_rng(12)
    idx = rng.permutation(110)
    tr_idx = idx[:100]
    te_idx = idx[100:]

    plot_training_iter(Phi_X_F[tr_idx], t[tr_idx], Phi_X_F[te_idx], t[te_idx], lamb=1e-2, lr=1e-6, fname='Wave_Plot_Train_Iter.png')
    #plot_func(X, t, lambda x: fourier_basis_func(x, 20, 120), LR, 'Function_Plot_GD_New.png')
    #plot_func(X, t, lambda x: np.concatenate([wave_basis_func(x, 5, 12), poly_basis_func((x.astype(int)//12), 3)[:, 1:]], axis=1), LR, 'Function_Plot_GD_Huber_New.png')