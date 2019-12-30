#%%
from functionOtimize import *
import argparse
clusteringMtd=[clusterAlgorithm,kMeansAlgorithm]

def readArgs():
    parser = argparse.ArgumentParser(description='Optimization and Machine Learning (MEEE-IPL).'
                                    ,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d',type=str,default=None,
    help='''Dataset Information: 
    1 - Dataset Path (.xlsx)
    2 - Number of rows to skip in dataset file 
    3 - Sheet Name''')
    parser.add_argument('-c',type=int,default=None,
    help='''Compute missing data: 
    No  : 0 
    Yes : 1''')
    parser.add_argument('-a',type=int,default=None,
    help='''Adapt Data:
    Normalize    : 0 
    Standardize  : 1''')
    parser.add_argument('-dt',type=str,default=None,
    help='''Pairwise Distance method availabe: braycurtis, canberra, chebyshev,
    cityblock, correlation, cosine, dice, euclidean, hamming, jaccard,
    jensenshannon, kulsinski, mahalanobis, matching, minkowski, rogerstanimoto, 
    russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule''')
    parser.add_argument('-ml',type=str,default=None,
    help="Linkage method available: single, complete, average, weighted, centeroid, median, ward")
    parser.add_argument('-nc' ,type=int,default=None,help="Number of Clusters")
    parser.add_argument('-cmt',type=int,default=None,help='''Select Clustering Method: 
    Cluster: 0
    K-Means: 1''')

    args = parser.parse_args()
    return args.d.split(" "),args.c,args.a,args.dt,args.ml,args.nc,args.cmt

[[datasetpath,numSkipedRow,sheetname],cmpMissData,adaptData,distanceMethod,linkageMethod,numCluster,cmt]=readArgs()

dataset=readExcel(datasetpath,int(numSkipedRow),sheetname)
[labels,data,dataDist,dataLink]=computeExcelData(dataset,cmpMissData,adaptData,distanceMethod,linkageMethod)
clusteringMtd[cmt](data,dataLink,numCluster)

plotBiDispersidade(data,None,0)
plotFunction(dendrogram,dataLink,labels=np.array(dataset.values)[:,0],title='Denogram', ylabel='Distance',xlabel='ID')


plt.show()
# %%prin

