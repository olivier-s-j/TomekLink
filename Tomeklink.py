from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Function that creates 2 clusters that partially overlap
# @return: X The datapoints e.g.: [f1, f2, ... ,fn]
# @return: y the classlabels e.g: [0,1,1,1,0,...,Cn]
def createCluster():
    X, y = make_blobs(n_samples=50, centers=2, n_features=2,random_state=0,center_box = (-5.0,5.0))
    return X.tolist(),y.tolist()

# Function which detects the Tomeklinks
# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @return: A 1-D array with the indices of the TomekLinks
def detectTomekLinks(X,y):
    tomeklinks = []
    neigh = NearestNeighbors(n_neighbors=30,algorithm = 'ball_tree')
    neigh.fit(X)

    for i in xrange(len(X) - 1, -1, -1):
        test = neigh.kneighbors(X[i],2,False)

        if y[i] != y[test[0][1]]:
            tomeklinks.append(test[0][1])
            tomeklinks.append(i)
    
    return tomeklinks

# @param: X The datapoints e.g.: [f1, f2, ... ,fn]
# @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
# @param: A 1-D array with the indices of the TomekLinks
# @return: X The datapoints e.g.: [f1, f2, ... ,fn]
# @return: y the classlabels e.g: [0,1,1,1,0,...,Cn]
def removeTomekLinks(tomeklinks,X,y):
    X = [i for j, i in enumerate(X) if j not in np.unique(tomeklinks)]
    y = [i for j, i in enumerate(y) if j not in np.unique(tomeklinks)]
    
    return X,y


