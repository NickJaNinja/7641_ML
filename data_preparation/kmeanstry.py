# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt 
import os


class Sklearn_Kmeans:
    def __init__(self, k):
        """
        Args: 
            
        """
        self.k = k


    def get_clustered_data(self, data):
        """
        """
        print(data.shape)
        k_means = KMeans(n_clusters=self.k) 
        k_means.fit(data) 
        clusters = k_means.predict(data)
        return clusters

    def process_images(self, source):
        x_train=[]
        source=source+'c1/'
        for i in range(10):
            source=source[:-2]+str(i)+'/'
            print(source)
            print(os.listdir(source))
            for file in os.listdir(source):
               print(2)
               image_file = os.path.join(source, file)
               print(image_file)
               img0=Image.open(image_file,'r')
               x_train.append(np.array(img0))
        x_train=np.array(x_train)
        cluster=self.get_clustered_data(x_train)
        return x_train, cluster
        
    def visualize_kmeans(self, data, clusters):
        for i in range(0,self.k):
            row = np.where(clusters==i)[0]  
            num = row.shape[0]       
            r = np.floor(num/10.)     

            print("cluster "+str(i))
            print(str(num)+" elements")

            plt.figure(figsize=(10,10))
            for k in range(0, num):
                plt.subplot(r+1, 10, k+1)
                image = data[row[k], ]
                image = image.reshape(8, 8)
                plt.imshow(image)
                plt.axis('off')
                plt.show()
        