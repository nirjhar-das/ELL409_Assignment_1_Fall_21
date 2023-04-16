from utils import fourier_basis_func
from wave_data_fitting import postprocess, preprocess
import numpy as np
import pandas as pd
from plot_utils import training_cross_val
from LinReg import LinearRegression

def grid_search(X, t, M_arr, lamb_arr):
    best_val_rmse = np.inf
    best_train_rmse = 0.0
    best_params = {'M': 0, 'lamb': 0}

    i = 1
    for M in M_arr:
        Phi_X = fourier_basis_func(X, M, 120)
        for lamb in lamb_arr:
            print("Current params : {}".format({'M': M, 'lamb': lamb}))
            train_rmse, curr_rmse, _ = training_cross_val(Phi_X, t, None, lamb, method='pinv')
            if curr_rmse < best_val_rmse:
                best_val_rmse = curr_rmse
                best_train_rmse = train_rmse
                best_params['M'] = M
                best_params['lamb'] = lamb
            print("{} searches done, current best rmse={}, params={}".format(i, best_val_rmse, best_params))
            i = i + 1
    return best_params, best_val_rmse, best_train_rmse

if __name__ == '__main__':
    df_train = pd.read_csv('train.csv')
    preprocess(df_train)
    df_test = pd.read_csv('test.csv')
    preprocess(df_test)
    M_arr = np.array([10, 20, 25, 30, 40, 50, 60]) # 6 val
    lamb_arr = np.power(10.0, np.arange(-10.0, 1.0, 1)) # 3 val
    
    X_train = np.array(df_train['n'])
    X_test = np.array(df_test['n'])
    t_train = np.array(df_train['value'])

    best_params, best_rmse, best_tr_rmse = grid_search(X_train, t_train, M_arr, lamb_arr)
    print('Best: {}'.format(best_params))
    print('10-CV RMSE: Train={}, Test={}'.format(best_tr_rmse, best_rmse))

    #best_params = {'M': 1000, 'k': 365, 'p': 2, 'q': 2, 'lamb': 1e+1}

    Phi_X_train = fourier_basis_func(X_train, best_params['M'], 120)
    Phi_X_test = fourier_basis_func(X_test, best_params['M'], 120)


    LR = LinearRegression(method='pinv', batch_size=1, learning_rate=1e-4, lambd=best_params['lamb'])

    #err = training_cross_val(Phi_X_train, t_train, method='gd', lamb=1e-2, bs=1)
    #print(err)

    LR.fit(Phi_X_train, t_train)
    print(LR.loss, LR.rmse)

    y_hat = LR.predict(Phi_X_test)
    date_list = df_test['id']

    postprocess(y_hat, date_list, output_fname='GD_Fourier.csv')
