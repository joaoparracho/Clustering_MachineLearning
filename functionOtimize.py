import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform #funções pdist e square form deve ser obtidas a partir do package scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
import skfuzzy  as  fuzz

def autolabel(rects,ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plotBar(data,**keyParam):
    fig,ax=plt.subplots()
    rects1=ax.bar(np.arange(len(data)),data)
    plt.xlabel(keyParam.pop('xlabel'))
    plt.ylabel(keyParam.pop('ylabel'))
    plt.title(keyParam.pop('title'))
    autolabel(rects1,ax)
    fig.tight_layout()
    plt.show(block=False)

def plotFunction(function,*positionalParm,**keyParam):
    plt.figure()
    plt.xlabel(keyParam.pop('xlabel'))
    plt.ylabel(keyParam.pop('ylabel'))
    plt.title(keyParam.pop('title'))
    function(*positionalParm,**keyParam)
    plt.show(block=False)

def plotBiDispersidade(data,cluster,i):
    plt.figure()
    plt.scatter(data[:,i],data[:,i+1],c=cluster,cmap='viridis',s=50, label='True  Position')#para  permitir  um  gráfico  de  dispersão  dos dados bidimensionais

    for label, x, y in zip(range(1,len(data)), data[:, 0], data[:, 1]):  # procedimento para incluir etiquetas junto aos pontos 
        plt.annotate(label, xy=(x, y),textcoords='offset points', ha='right', va='bottom')
    
    plt.grid() # para integrar uma grelha
    plt.show(block=False) 

def readExcel(excelPath,numSkipedRow=0,sheetName="Clustering"):
    return pd.read_excel(excelPath,sheetName,skiprows=numSkipedRow)

def computeExcelData(excelDataSet,cmpMissData=1,adaptData=1,distanceMethod="euclidean",linkageMethod="average"):
    excelDataSet.fillna(excelDataSet.mean(),inplace=True) if cmpMissData else  excelDataSet.dropna(inplace=True) 
    data=preprocessing.StandardScaler().fit_transform(np.array(excelDataSet.values)[:,1:]) if adaptData else preprocessing.normalize(np.array(excelDataSet.values)[:,1:])
    dataDist=pdist(data,distanceMethod)
    dataLink=linkage(dataDist,linkageMethod)
    print(excelDataSet.isnull().sum())
    print(squareform(dataDist))
    return excelDataSet.columns.values,data,dataDist,dataLink

def clusterAlgorithm(data,dataLink,numCluster):
    C=fcluster(dataLink,numCluster,'maxclust')
    centeroid=np.zeros([numCluster,len(data[0])])
    for i in range(1,numCluster+1):
        centeroid[i-1,:]=data[C==i,:].mean(axis=0)
    print(centeroid)


def kMeansAlgorithm(data,dataLink,numCluster):
    
    Sum_of_squared_distances  =  []
    for k in range(2,10): # para permitir uma análise para umnúmero variável de clusters
        km=KMeans(n_clusters=k).fit(data)
        Sum_of_squared_distances.append(km.inertia_) #  km.inertia  representaa  soma quadrática das distâncias no interior de cada cluster
        silhouette_avg  =  silhouette_score(data,  km.labels_) # para  obtenção  dos  coeficientes  de Silhouette (media dos valores de Silhouette)
        print("For n_clusters =", k, "The average silhouette_scoreis :", silhouette_avg)
        silhouette_values=silhouette_samples(data,km.labels_) # para   obtenção   dovalor   de Silhouette para cada objeto
        print("For n_clusters =", k, "silhouette_values are :", silhouette_values)
    

    km = KMeans(n_clusters=numCluster).fit(data)#o comando kmeans permite o recurso ao método de clustering K-means, de forma a agrupar as observações identificadas por cada linha de A em N clusters (neste caso 3 clusters foram definidos). 
    centroids=km.cluster_centers_

    plotFunction(plt.plot,range(2,10),Sum_of_squared_distances, 'bx-',title='Elbow Method For Optimal k', ylabel='Sum_of_squared_distances',xlabel='k')
    
    for i in range(0,len(data[0]),2):
        plotBiDispersidade(data, km.predict(data),i)
        plt.scatter(centroids[:,i],centroids[:,i+1],c='black',s=200,alpha=0.5)
    
    #plt.show(block=False) 
    numObjCluster=np.zeros([km.predict(data).max()+1])

    for k in range(0,km.predict(data).max()+1):
        numObjCluster[k]=(km.predict(data)==k).sum()

    print(numObjCluster)
    plotBar(numObjCluster.astype(int),title='Numbero of objects per CLuster', 
    ylabel='Number of Object',xlabel='Cluster Index')

def fuzzyCmeans(data,dataL,numCluster):
    cntr, u, u0, d, ObjFunction, p, fpc = fuzz.cluster.cmeans(data.T, 3, 2, error=0.005,maxiter=100, init=None)
    cntr,  u,  u0,  d,  ObjFunction,  numiter,  fpc  =  fuzz.cluster.cmeans(data.T,  2,  2, error=0.005, maxiter=1000, init=None)
    u_pred, u0_pred, d_pred, ObjFunction_pred, numiter_pred, fpc_pred= fuzz.cluster.cmeans_predict(dataL,   cntr,2,   error=0.005,   maxiter=1000,init=None)
