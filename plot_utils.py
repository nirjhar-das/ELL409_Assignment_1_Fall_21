from LinReg import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from utils import poly_basis_func
from sklearn.model_selection import KFold
import pandas as pd

def training_cross_val(Phi_X, t, bs, lamb, p=2, q=2, lr=1e-2, method='gd', beta=0.9, rms=0.999, n_splits=10):
    kf = KFold(n_splits=n_splits, random_state=23, shuffle=True)
    train_loss, train_rmse_list, test_rmse_list = [], [], []
    for train_idx, test_idx in kf.split(Phi_X):
        Xtr, Ytr = Phi_X[train_idx], t[train_idx]
        Xte, Yte = Phi_X[test_idx], t[test_idx]
        LR = LinearRegression(method=method, batch_size=bs, lambd=lamb, learning_rate=lr, p=p, q=q, beta=beta, rms=rms)
        LR.fit(Xtr, Ytr)
        train_loss.append(LR.loss)
        train_rmse_list.append(LR.rmse)
        test_rmse_list.append(LR.evaluate(Xte, Yte))

    return np.mean(train_rmse_list), np.mean(test_rmse_list), np.mean(train_loss)

def plot_poly_degree(X, t, fname, lamb=1e-10, method='pinv'):
    train_rmse, test_rmse = [], []
    for m in range(1, 16):
        Phi_X = poly_basis_func(X, m)
        tr_rmse, te_rmse, _ = training_cross_val(Phi_X, t, lamb=lamb, p=2, q=2, method=method, bs=100)
        train_rmse.append(tr_rmse)
        test_rmse.append(te_rmse)
    
    m_arr = np.arange(1, 16, 1)

    plt.plot(m_arr, train_rmse, label='Train')
    plt.plot(m_arr, test_rmse, label='Test')
    plt.title('RMSE vs Degree')
    plt.legend()
    plt.savefig(fname)
    plt.show()

def plot_training_lambda(Phi_X, t, fname, method='pinv'):
    train_rmse, test_rmse = [], []
    lamb_arr = np.power(10.0, np.arange(-15.0, 2.0, 1.0))
    for l in lamb_arr:
        tr_rmse, te_rmse, _ = training_cross_val(Phi_X, t, bs=100, lamb=l, method=method)
        train_rmse.append(tr_rmse)
        test_rmse.append(te_rmse)
    
    log_lamb = np.log10(lamb_arr)
    fig, ax = plt.subplots(1)

    ax.plot(log_lamb, train_rmse, label='Train')
    ax.plot(log_lamb, test_rmse, label='Test')
    ax.legend()
    ax.set_title('RMSE vs Log-Lambda')

    fig.tight_layout()
    plt.savefig(fname)
    plt.show()

def plot_training_iter(Phi_X_train, t_train, Phi_X_test, t_test, lamb, lr, fname):
    LR = LinearRegression(method='gd', batch_size=100, lambd=lamb, learning_rate=lr, beta=0.9, rms=0.999)
    train_loss, train_rmse, test_rmse = LR.fit(Phi_X_train, t_train, Phi_X_test, t_test, verbosity='list', tol=0.0, max_iter=20000)
    
    epoch_arr = np.arange(1, len(train_loss)+1, 1)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(epoch_arr, train_loss)
    ax[0].set_title('Training Loss vs Num_Epochs')

    ax[1].plot(epoch_arr, train_rmse, label='Train')
    ax[1].plot(epoch_arr, test_rmse, label='Test')
    ax[1].legend()
    ax[1].set_title('RMSE vs Num_Epochs')

    fig.tight_layout()
    plt.savefig(fname)
    plt.show()

def plot_training_batch(Phi_X, t, lamb, fname):
    train_loss, train_rmse, test_rmse = [], [], []
    bs_arr = np.arange(0, Phi_X.shape[0] + 1, Phi_X.shape[0]//10)
    bs_arr[0] += 1
    for bs in bs_arr:
        tr_rmse, te_rmse, loss = training_cross_val(Phi_X, t, bs, lamb=lamb)
        train_loss.append(loss)
        train_rmse.append(tr_rmse)
        test_rmse.append(te_rmse)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(bs_arr, train_loss)
    ax[0].set_title('Training Loss vs Batch Size')

    ax[1].plot(bs_arr, train_rmse, label='Train')
    ax[1].plot(bs_arr, test_rmse, label='Test')
    ax[1].legend()
    ax[1].set_title('RMSE vs Batch Size')

    fig.tight_layout()
    plt.savefig(fname)
    plt.show()

def plot_func(X, t, basis_func, lr, fname):
    X_new = np.sort(np.concatenate((np.linspace(min(X), max(X), len(X)), X)))
    Phi_X_new = basis_func(X_new)
    y_new = lr.predict(Phi_X_new)

    plt.plot(X_new, y_new)
    plt.scatter(X, t, c='r')
    plt.savefig(fname)
    plt.show()

def training_loss_func(Phi_X, t, lamb, fname):
    p_arr = [0, 1, 2]
    q_arr = [1, 2]
    data = {'p': [], 'q': [], 'train_rmse': [], 'test_rmse': []}
    for p in p_arr:
        for q in q_arr:
            tr_rmse, te_rmse, _ = training_cross_val(Phi_X, t, bs=100, lamb=lamb, p=p, q=q)
            data['p'].append(p)
            data['q'].append(q)
            data['train_rmse'].append(tr_rmse)
            data['test_rmse'].append(te_rmse)
    df = pd.DataFrame(data=data)
    df.to_csv(fname, index=False)

def calc_noise(Phi_X, t, LR, fname):
    ypred = LR.predict(Phi_X)
    noise = (t - ypred).squeeze()
    var = np.var(noise)
    mu = np.mean(noise)
    #print(noise)
    plt.hist(noise, bins=25, label='Mean: {}\nVariance: {}'.format(mu, var))
    plt.legend()
    plt.title('Noise Distribution')
    plt.savefig(fname)
    plt.show()