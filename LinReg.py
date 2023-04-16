from utils import gradient_fn, huber_grad_fn, huber_loss_fn, loss_fn
import numpy as np

class LinearRegression:
    def __init__(self, method='pinv', batch_size=None, p=2, q=2, learning_rate=1e-4, beta=0.9, decay=0, rms=0.999, lambd=0):
        self.w = None
        self.grad = None
        self.v = None
        self.method = method
        self.p = p
        self.q = q
        self.bs = batch_size
        self.lr = learning_rate
        self.beta = beta
        self.lamb = lambd
        self.loss = 0.0
        self.rmse = 0.0
        self.decay = decay
        self.rms = rms
        self.rng = np.random.default_rng(4771)
  
    def fit(self, Phi_X, t, Phi_X_te=None, t_te=None, tol=1.0e-5, max_iter=5000, verbosity=0):
        if self.w is None:
            self.w = self.rng.normal(size=(Phi_X.shape[1], 1))
        if self.method == 'pinv':
            self.w = np.linalg.inv(Phi_X.T @ Phi_X + self.lamb*np.eye(Phi_X.shape[1])) @ Phi_X.T @ t
            y_hat = Phi_X @ self.w
            self.loss = loss_fn(y_hat, t, self.w, self.lamb, self.p, self.q)
            self.rmse = np.sqrt(np.mean((y_hat - t)**2))
            if verbosity != 'list' and verbosity > 0:
                print('Loss: {}'.format(self.loss))
                print('RMSE: {}'.format(self.rmse))

        elif self.method == 'gd':
            if verbosity == 'list':
                loss_arr, rmse_arr = [], []
                if not Phi_X_te is None:
                    test_arr = []
            N = Phi_X.shape[0]
            if self.grad is None:
                self.grad = np.zeros_like(self.w)
            if self.rms != 0.0:
                self.v = np.zeros_like(self.w)
            idx = self.rng.permutation(N)
            for j in range(max_iter):
                if j > 0:
                    self.lr = self.lr*(1 + self.decay*(j-1))/(1 + self.decay*j)
                Phi_X = Phi_X[idx]
                t = t[idx]
                n_batches = N // self.bs + (N % self.bs != 0)
                for i in range(n_batches):
                    xbatch = Phi_X[i*self.bs : min((i+1)*self.bs, N)]
                    tbatch = t[i*self.bs : min((i+1)*self.bs, N)]
                    y_hat = xbatch @ self.w
                    if self.p == 0:
                        del_w = huber_grad_fn(xbatch, y_hat, tbatch, self.w, self.lamb, self.q)
                    else :
                        del_w = gradient_fn(xbatch, y_hat, tbatch, self.w, self.lamb, self.p, self.q)
                    self.grad = (self.beta * self.grad) + ((1 - self.beta) * del_w)
                    grad_p = self.grad/(1.0 - self.beta**(j+1))
                    if self.rms > 0.0:
                        self.v = self.rms*self.v + (1 - self.rms)*(self.grad**2)
                        v_p = self.v/(1.0 - self.rms**(j+1))
                        self.w = self.w - self.lr*(grad_p/(1e-8 + np.sqrt(v_p)))
                    else:
                        self.w = self.w - self.lr*self.grad


                y_hat = Phi_X @ self.w
                if self.p == 0:
                    curr_loss = huber_loss_fn(y_hat, t, self.w, self.lamb, self.q)
                else :
                    curr_loss = loss_fn(y_hat, t, self.w, self.lamb, self.p, self.q)
                self.rmse = np.sqrt(np.mean((y_hat - t)**2))

                if verbosity == 'list':
                    loss_arr.append(curr_loss)
                    rmse_arr.append(self.rmse)
                    if not Phi_X_te is None:
                        test_arr.append(self.evaluate(Phi_X_te, t_te))
                
                if verbosity != 'list' and verbosity > 0:
                    if j%verbosity == 0:
                        print('Loss in {}-th iteration: {}'.format(j, curr_loss))
                        print('RMSE in {}-th iteration: {}'.format(j, self.rmse))
                        if not Phi_X_te is None:
                            print('Test RMSE in {}-th iteration: {}'.format(j, self.evaluate(Phi_X_te, t_te)))
                
                if self.loss != 0.0 and abs(self.loss - curr_loss)/self.loss < tol:
                    self.loss = curr_loss
                    if verbosity == 'list':
                        if Phi_X_te is None:
                            return loss_arr, rmse_arr
                        else:
                            return loss_arr, rmse_arr, test_arr
                    else :
                        return
                else:
                    self.loss = curr_loss
            
            if verbosity == 'list':
                if Phi_X_te is None:
                    return loss_arr, rmse_arr
                else:
                    return loss_arr, rmse_arr, test_arr
        
        else:
            return
    
  
    def predict(self, Phi_X):
        return Phi_X @ self.w

    def evaluate(self, Phi_X, t):
        y_hat = Phi_X @ self.w
        rmse = np.sqrt(np.mean((y_hat - t)**2))
        return rmse