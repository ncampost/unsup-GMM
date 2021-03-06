import numpy as np
from scipy.stats import multivariate_normal
from KM import KMeans
import math

class GaussianMixtureModel:
    def __init__(self):
        return
    
    # Returns Q, a N x K matrix, where Qnk is the probability
    # that X_n belongs to the kth class.
    def fit(self, X, K):
        # init params
        N = X.shape[0]
        D = X.shape[1]

        # Initialize {Q}, {Pi, Mu, Sigma}.
        # Here we run KMeans to cluster and use the result as an initialization of {Q},
        # and {Pi, Mu, Sigma} are implicit from the resulting {Q}.
        KM = KMeans()
        Z = KM.fit(X, K)
        Q = np.zeros((N, K))
        Q[np.arange(N), Z] = 1
    
        # Perform coordinate ascent until difference 
        # in log-likelihood is small (here: < 1e-1)
        init = False
        p = 0
        while True:
            # Update optimal {Pi, Mu, Sigma} given fixed {Q}
            sum_n_qnk = np.sum(Q, axis=0)
            Pi = sum_n_qnk / N
            Mu = np.dot(X.T, Q) / sum_n_qnk
            Sigma = np.zeros((K, D, D))
            for k in range(K):
                for n in range(N):
                    XMu = np.reshape(X[n]-Mu[:,k], (D, 1))
                    Sigma[k] += Q[n,k]*np.dot(XMu, XMu.T)
                Sigma[k] /= sum_n_qnk[k]
            
            # Check whether we have achieved stopping criterion
            if not init:
                LL = self.__calc_LL(X, Q, Pi, Mu, Sigma)
                init = True
            else:
                oldLL = LL
                LL = self.__calc_LL(X, Q, Pi, Mu, Sigma)
                if math.fabs(LL - oldLL) < 1e-1:
                    break

            # Update optimal {Q} given fixed {Pi, Mu, Sigma}
            for n in range(N):
                for k in range(K):
                    Nprob = multivariate_normal.pdf(X[n], mean=Mu[:,k], cov=Sigma[k])
                    Q[n,k] = Pi[k] * Nprob
                Q[n] /= np.sum(Q[n])

        return Q
    
    # Calculate negative log likelihood
    def __calc_LL(self, X, Q, Pi, Mu, Sigma):
        N = Q.shape[0]
        K = Q.shape[1]
        sum = 0
        for n in range(N):
            for k in range(K):
                sum += Q[n,k]*math.log(Pi[k])
                sum += Q[n,k]*multivariate_normal.logpdf(X[n], mean=Mu[:,k], cov=Sigma[k])
        return sum
    
    # Softmax a row vector
    def __softmax(self, row):
        sum = np.sum(np.exp(row))
        return np.exp(row) / sum