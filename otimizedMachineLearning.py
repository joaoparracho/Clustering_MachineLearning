#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform #funções pdist e square form deve ser obtidas a partir do package scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import skfuzzy  as  fuzz 
import sys

#datasetpath=sys.argv[1]
datasetpath="D:\\IPL\\Mestrado\\1_Ano\\OML\\Projecto\\Clustering_MachineLearning\\Datasets_GroupG.xlsx"
datasetUnsupervised=pd.read_excel(datasetpath,skiprows=1)
labelsUnsupervised=datasetUnsupervised.columns.values
print(labelsUnsupervised)
print(datasetUnsupervised)
#print(dataset.values)
#data=dataset.values[1:,:]
#print(dataset.isnull().sum())
#print(dataset)
#dataset.fillna(dataset.mean(), inplace=True)
# count the number of NaN values in each column
#print(dataset.isnull().sum())
# %%
