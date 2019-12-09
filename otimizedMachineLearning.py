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
dataset=np.array(pd.read_excel(datasetpath)).T
print(dataset)

# %%
