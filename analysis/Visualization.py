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





def HOGplot(data,hog,n=3):
    '''
    Args:
        data: N x H x W x D (N images before processed by HOG.py)
        
        hog: output of HOG.py, which is a tuple of three numpy arrays: (extracted_features, images_before_hog, images_after_hog)
        
        n: number of image comparisons - by default you will get at most 3 pairs of before and after images
    
    Returns:
        images comparison before and after HOG
    
    '''
    
    _,h = hog
    
    N = data.shape[0]
    
    if N <= n:
        n = N

    for i in range(n):
    
        orig_image = data[i]
  
        hog_image = h[i]
      

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(orig_image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()