
import numpy as np
import random

class KMeans:
    def __init__(self):
        return
    
	def fit(self, X, K):
		# init params
		N = X.shape[0]
		D = X.shape[1]
		z = np.zeros(N, dtype=int)
	
		# Random restart: choose each mean to be a random member of X
		means = np.zeros((K, D))
		S = []
		for i in range(K):
			randint = random.randint(0, N - 1)
			while randint in S:
				randint = random.randint(0, N - 1)
			S.append(randint)
			means[i] = X[randint]
		del S

		# Perform coordinate ascent in {means}, {z} until
		# algorithm has converged (there were no changes in {z})
		while True:
			j = 0
			changed = False
			for datum in X:
				argmin = 0
				mindist = 0
				i = 0 
				init = False
				# TODO: use a more efficient structure to store distances as opposed
				# to recalculating them every time.
				for mean in means:
					dist = np.linalg.norm(datum - mean)
					if not init:
						argmin = i
						mindist = dist
						init = True
					elif dist < mindist:
						argmin = i
						mindist = dist
						i += 1

				# Reassign point's class to the closest mean
				if z[j] != argmin:
					changed = True
					z[j] = argmin
				j += 1
			
			# If no datum had its class reassigned, we have converged.
			if not changed:
				break
			
			for i in range(K):
				# Recalculate means of classes.
				num_assigned = 0
				sum_matr = np.zeros((1,D))
				for j in range(N):
					if z[j] == i:
						sum_matr += X[j]
						num_assigned += 1
				sum_matr /= num_assigned
				means[i] = sum_matr
        
		# Return cluster classes.
		return z
