import argparse
import numpy as np  
from LinReg import LinearRegression
from utils import poly_basis_func


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=1, help="part of question")  
    parser.add_argument("--method", default="pinv", help = "type of solver")  
    parser.add_argument("--batch_size", default=5, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=float, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()
    

if __name__ == '__main__':
    args = setup()
    LR = LinearRegression(method=args.method, batch_size=args.batch_size, lambd=args.lamb, learning_rate=1e-2)
    data = np.genfromtxt(args.X, delimiter=',')
    X, t = data[:, 0], data[:, 1]
    Phi_X = poly_basis_func(X, int(args.polynomial))
    LR.fit(Phi_X, t)
    print("weights={}".format(LR.w.squeeze()))
    