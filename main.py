# Model
import numpy as np
from GMM import GaussianMixtureModel

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Data generator
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_classification

def main():
    hidden_classes = 2
    model_classes = 2
    #X, _ = make_blobs(n_samples=700, centers=hidden_classes)
    #X, _ = make_circles(n_samples=700)
    X, _ = make_classification(n_samples=600, n_features=2, n_informative=2, n_redundant=0, n_classes=hidden_classes)

    # Init and fit Gaussian Mixture Model
    GMM = GaussianMixtureModel()
    Q = GMM.fit(X, model_classes)
    

    # Visualize data where we assign each X_n to
    # the class it has highest probability.
    vis_data_shade(X, Q, model_classes, hidden_classes)

# Shades each point proportional to the probability that the point
# belongs to that class.
def vis_data_shade(X, Q, n_classes, hidden_classes):
    # Use specific RGB values instead of discrete char identifiers if num_classes > 8
    char_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    rgb_colors = []
    for i in range(n_classes):
        rgb_colors.append(c.to_rgb(char_colors[i]))
    rgb_colors = np.array(rgb_colors).flatten()
    colorings = np.zeros((Q.shape[0], 3))
    for n in range(Q.shape[0]):
        for k in range(Q.shape[1]):
            # Calculate R,G,B contributions
            colorings[n,0] += Q[n,k]*rgb_colors[k*3]
            colorings[n,1] += Q[n,k]*rgb_colors[k*3+1]
            colorings[n,2] += Q[n,k]*rgb_colors[k*3+2]
    

    plt.scatter(X[:,0], X[:,1], c=colorings, edgecolor='black')
    plt.title("Hidden classes: " + str(hidden_classes) + ". Modeled classes: " + str(n_classes) + ".")
    plt.show()

# Vis_data for discrete class assignments.
def vis_data(X, Z):
    # Use specific RGB values instead of discrete char identifiers if num_classes > 8
    cMap = c.ListedColormap(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'])
    plt.scatter(X[:,0], X[:,1], c=Z, cmap=cMap, edgecolor='black')
    plt.show()

if __name__ == '__main__':
    main()