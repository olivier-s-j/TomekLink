import matplotlib.pyplot as pl
import Tomeklink
import numpy as np


# Test code to visually verify that the TomekLinks are removed.
def main():
    
    # Create 2 artificial clusters that partially overlap
    X,y = Tomeklink.createCluster()
    
    # Plot the clusters
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    pl.scatter(np.array(X)[:, 0], np.array(X)[:, 1], color=colors[y].tolist(), s=10)
    pl.show()

    # Detect the TomekLinks in the data
    tomeklinks = Tomeklink.detectTomekLinks(X,y)
    
    # Remove the TomekLinks from the data
    X,y = Tomeklink.removeTomekLinks(tomeklinks,X,y)
     
    # Plat the data again with the TomekLinks removed
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    pl.scatter(np.array(X)[:, 0], np.array(X)[:, 1], color=colors[y].tolist(), s=10)
    pl.show()


if  __name__ =='__main__':main()