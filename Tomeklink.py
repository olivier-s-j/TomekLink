
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Function that creates 2 clusters that partially overlap
# @return: X The datapoints e.g.: [f1, f2, ... ,fn]
# @return: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @Note : X and y are arrays
def createCluster():
    X, y = make_blobs(n_samples=500, centers=2, n_features=2,random_state=0,center_box = (-5.0,5.0))
    return X,y

# Function which detects the Tomeklinks
# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @return: 1-D arrays with the indices of the TomekLinks and others

def detectTomekLinks(X,y):
    nonlinks = []
    neigh = NearestNeighbors(n_neighbors=2,algorithm = 'kd_tree')
    neigh.fit(X)

    # k2 stores the first nearest neighbour
    k2 = neigh.kneighbors(X)[1]

    # k_tomek stores the ones where the labels conflict.
    k_tomek = k2[y != y[k2[:,1]]]

    # This is for getting the positions
    tomekList = np.unique(np.concatenate([k_tomek[:,0],k_tomek[:,1]]))
    index = np.arange(0,len(X))
    nonlinks = set(index) - set(tomekList)
    nonlinks = list(nonlinks)
    return np.asarray(nonlinks), np.array(tomekList)


# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @param: A 1-D array with the indices of the TomekLinks
# @return: X The datapoints e.g.: [f1, f2, ... ,fn]
# @return: y the classlabels e.g: [0,1,1,1,0,...,Cn]
def removeTomekLinks(X,y,tomeklinks):
    return X[tomeklinks], y[tomeklinks]


# a quick  test
if __name__ == "__main__":

    X,y = createCluster()
    nonlinks,tomeklinks = detectTomekLinks(X,y)
    X_clean,y_clean = removeTomekLinks(X,y,tomeklinks)

