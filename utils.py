import enum
import numpy as np

def poly_basis_func(X, deg=5):
    Phi_X = np.zeros((X.shape[0], deg+1))
    for i in range(deg+1):
        Phi_X[:, i] = X**i
  
    return Phi_X

def wave_basis_func(X, M, k):
    w0=2.0*np.pi/float(k)
    N = X.shape[0]
    Phi_X = np.zeros((N, 2*M + 1))
    Phi_X[:, 0] = 1
    for i in range(1, 2*M + 1):
        if (i%2 != 0):
            Phi_X[:, i] = np.cos(w0*i*X)
        else :
            Phi_X[:, i] = np.sin(w0*i*X)
    
    return Phi_X

def exp_basis_func(X, mu, s):
    N = X.shape[0]
    M = mu.shape[0]
    Phi_X = np.zeros((N, M))
    for i in range(M):
        Phi_X[:, i] = np.exp(-((X - mu[i])**2)/(2*(s**2)))
    
    return Phi_X

def fourier_basis_func(X, M, k):
    w0=2.0*np.pi/float(k)
    N = X.shape[0]
    Phi_X = np.zeros((N, 2*M + 1))
    Phi_X[:, 0] = 1
    for i in range(1, 2*M + 1):
        if (i%2 != 0):
            Phi_X[:, i] = np.cos(w0*(i//2 + 1)*X)
        else :
            Phi_X[:, i] = np.sin(w0*(i//2)*X)

    return Phi_X

def custom_basis(X, freq):
    Phi_X = np.zeros((X.shape[0], 2*len(freq) + 1))
    Phi_X[:, 0] = 1

    #for i range(1, 2*len(freq)):
    #    Phi_X[:, i+1] = 


def loss_fn(y, t, w, lamb, p=2, q=2):
    return 0.5*np.mean(np.abs(y - t)**p) + 0.5*lamb*np.sum(np.abs(w)**q)

def huber_loss_fn(y, t, w, lamb, delta=1.0, q=2):
    l1 = np.mean(np.where(np.abs(y - t) <= delta, 0.5*(y-t)**2, delta*(np.abs(y - t) - 0.5*delta)))
    l2 = 0.5*lamb*np.sum(np.abs(w)**q)
    return l1 + l2

def huber_grad_fn(x, y, t, w, lamb, delta=1.0, q=2):
    tol = 1.0e-8
    t = t.reshape(-1, 1)
    g1 = np.mean(x*np.where(np.abs(y - t) <= delta, y - t, delta*np.sign(y - t)), axis=0).reshape(-1, 1)
    g2 = 0.5*q*lamb*np.sign(w)*(np.where(np.abs(w) < tol, 0, np.abs(w)**(q-1))).reshape(-1, 1)
    return g1 + g2

def gradient_fn(x, y, t, w, lamb, p=2, q=2):
    #tol = 1.0e-8
    t = t.reshape(-1, 1)
    #g1 = 0.5*p*np.mean(x* np.where(np.abs(y - t) < tol, 0, np.sign(y - t)*((np.abs(y - t)**(p-1)))), axis=0).reshape(-1, 1)
    g1 = 0.5*p*((x.T) @ (np.sign(y - t)*(np.abs(y - t)**(p-1)))).reshape(-1, 1)
    #g1 = (x.T @ (y-t)).reshape(-1, 1)
    #g2 = 0.5*q*lamb*np.sign(w)*(np.where(np.abs(w) < tol, 0, np.abs(w)**(q-1))).reshape(-1, 1)
    g2 = 0.5*q*lamb*np.sign(w)*(np.abs(w)**(q-1))
    return g1 + g2

def part1_preprocess(fname):
    data = np.genfromtxt(fname, delimiter=',')
    X, t = data[:, 0], data[:, 1]

    return X, t


