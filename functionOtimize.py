import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform #funções pdist e square form deve ser obtidas a partir do package scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
import skfuzzy  as  fuzz 

def plotFunction(function,*positionalParm,**keyParam):
    plt.figure
    function(*positionalParm,**keyParam)
    plt.show(block=False)

def readExcel(excelPath,numSkipedRow=0,sheetName="Clustering"):
    return pd.read_excel(excelPath,sheetName,skiprows=numSkipedRow)

def computeExcelData(excelDataSet,cmpMissData=1,adaptData=1,distanceMethod="euclidean",linkageMethod="average"):
    labels=excelDataSet.columns.values
    excelDataSet.fillna(excelDataSet.mean(),inplace=True) if cmpMissData else  excelDataSet.dropna(inplace=True) 
    data=preprocessing.StandardScaler().fit_transform(np.array(excelDataSet.values)[:,1:]) if adaptData else preprocessing.normalize(np.array(excelDataSet.values)[:,1:])
    dataDist=pdist(data,distanceMethod)
    dataLink=linkage(dataDist,linkageMethod)
 
    print(excelDataSet.isnull().sum())
    print(squareform(dataDist))
    return labels,data,dataDist,dataLink

def clusterAlgorithm(data,dataLink,numCluster):
    C=fcluster(dataLink,numCluster,'maxclust')
    for i in range(1,numCluster):
        centroid=data[C==i,:].mean(axis=0)
    print(centroid)


def kMeansAlgorithm(data,dataLink,numCluster):
    print("Heu")

