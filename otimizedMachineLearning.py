#%%
from functionOtimize import *
import argparse
clusteringMtd=[clusterAlgorithm,kMeansAlgorithm]

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
