import matplotlib.pyplot as plt


def PCAplot2d(pca):
    '''
    Parameters
    ----------
    pca : TYPE
        This will make a 2D pca plot if you run with the output of "get_reduced_data" from Sklearn_PCA.py.

    Returns 2D plot
    -------

    '''
    plot = plt.scatter(pca[:,0], pca[:,1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("First two principal components")
    plt.show()

