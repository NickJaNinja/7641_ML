#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:33:45 2022

@author: bshi42
"""

from kmeanstry import Sklearn_Kmeans

km=Sklearn_Kmeans(10)
data, cluster=km.process_images('/home/bshi42/Desktop/MLproject/imgs/train/')
cuts=[]
for i in range(10):
    len=250
    clip=cluster[i*len: (i+1)*len]
    cuts.append(clip)

import statistics as st
for i in cuts:
    i=list(i)
    mode=st.mode(i)
    print(mode)
    print(i.count(mode))
    print(i[1:10])
    
import matplotlib.pyplot as plt
plt.hist(cluster, bins=10)
