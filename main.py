# Model
import numpy as np
from GMM import GaussianMixtureModel

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Data generator
from sklearn.datasets.samples_generator import make_blobs

def main():
    hidden_classes = 3
    model_classes = 3
    X, Y = make_blobs(n_samples=300, centers=hidden_classes)

    # Init and fit Gaussian Mixture Model
    model = GaussianMixtureModel()
    Q = model.fit(X, model_classes)

    # Visualize data where we assign each X_n to
    # the class it has highest probability.
    vis_data(X, np.argmax(Q, axis=1))

# Simple vis_data. TODO: support shading where for each point
# we shade class colors proportional to the probability that
# the point belongs to that class.
def vis_data(x,y):
    # Add more colors here if model_classes > 4
    cMap = c.ListedColormap(['r', 'b', 'g', 'm'])
    plt.scatter(x[:,0], x[:,1], c=y, cmap=cMap, edgecolor='black')
    plt.show()

if __name__ == '__main__':
    main()