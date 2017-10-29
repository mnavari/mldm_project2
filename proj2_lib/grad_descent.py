import numpy as np
import math

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# initialize model parameters w and b
# intializing to 0 is not a good idea
# it should be a random vector see np.random.randn
def _initialize_parameters(nfeatures):
    sd = math.sqrt(0.5 * nfeatures)
    w = np.random.randn(nfeatures) / sd
    b = np.random.randn(1) / sd
    return w, b

# compute signed distances
# based on current model estimates w and b
def _get_signed_distances(X, w, b):
    f = np.dot(X, w) + b
    return f

# this is a vectorized version of positive_part operation
# we can use this for hinge loss as post_part(1.0 - y*f)
pos_part = np.vectorize(lambda u: u if u > 0. else 0.)

# compute the value of the linear SVM objective function
# given current signed distances, and parameter vector w
def _get_objective(f, y, w, lam):
    nobs = f.shape[0]
    loss = np.sum(pos_part(1. - y*f)) / nobs
    penalty = 0.5 * lam * np.dot(w,w)
    return loss + penalty

# compute gradients with respect to w and b
subgrad = np.vectorize(lambda yf: -1. if yf < 1. else 0.)

def _get_gradients(f, X, y, w, b, lam):
    nobs = X.shape[0]
    yf = y * f
    t = subgrad(yf)
    ty = t * y
    
    gw = np.sum(np.multiply(X.T, ty).T, axis=0) / nobs
    gw += lam * w
    
    gb = np.sum(ty) / nobs
    return gw, gb

# fit an SVM using gradient descent
# X: matrix of feature values
# y: labels (-1 or 1)
# n_iter: numer of iterations
# eta: learning rate
# tol: stopping condition tolerance, stop iterations if norm of gradient 
#      becomes smaller than this value
# verbose: print iteration information if > 0
def fit_svm(X, y, lam, n_iter=1000, eta=.5, tol=1e-10, verbose=0):
    nexamples, nfeatures = X.shape
    print_size = round(n_iter / 10)
    
    w, b = _initialize_parameters(nfeatures)
    
    # initialize these values
    gw = np.full((nfeatures), 0.0)
    gb = np.full((1), 0.0)
    
    for k in range(n_iter):
        f = _get_signed_distances(X, w, b)
        
        # print information and 
        # update the learning rate
        if k % print_size == 0:
            obj = _get_objective(f, y, w, lam)
            penalty = 0.5 * lam * np.dot(w,w)
            loss = obj - penalty
            grad_size = np.sum(gw**2) + gb**2 if k > 0 else np.inf
            eta = 0.5 * eta
            
            if verbose > 0:
                print("it: %d, obj %.2f, loss %.2f, g: %.2f" % (k, obj, loss, grad_size))
                
            # check stopping condition
            if grad_size < tol:
                break
        
        gw, gb = _get_gradients(f, X, y, w, b, lam)
        
        # update model parameters
        w = w - eta * gw
        b = b - eta * b
    return w, b

class GDLinearSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, lam=1.0, n_iter=1000, eta=0.5, tol=1e-10, verbose=0):
        self.lam = lam
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        self.verbose = verbose
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        self.w_, self.b_ = fit_svm(X, y, lam=self.lam, n_iter=self.n_iter, 
                                   eta=self.eta, verbose=self.verbose)
        
        return self
    
    def decision_function(self, X):
        check_is_fitted(self, ['w_', 'b_'])
        X = check_array(X)
        return _get_signed_distances(self, self.w_, self.b_)
    
    def predict(self, X):
        f = self.decision_function(X)
        return np.sign(f)
        