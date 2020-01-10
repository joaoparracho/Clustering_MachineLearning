#%%
#Check outliers

from functionOtimize import *
import argparse
clusteringMtd=[clusterHAlgorithm,kMeansAlgorithm,fuzzyCmeansAlgorithm]
normStr=["Min_Max","StandarScaler"]
def readArgs():
    parser = argparse.ArgumentParser(description='Optimization and Machine Learning (MEEE-IPL).'
                                    ,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d',type=str,default="Datasets_GroupG.xlsx 1 Clustering",
    help='''Dataset Information: 
    1 - Dataset Path (.xlsx)
    2 - Number of rows to skip in dataset file 
    3 - Sheet Name''')
    parser.add_argument('-c',type=int,default=0,
    help='''Compute missing data: 
    No  : 0 
    Yes : 1''')
    parser.add_argument('-a',type=int,default=0,
    help='''Adapt Data:
    Normalize (Min_Max) : 0 
    Standardize         : 1''')
    parser.add_argument('-dt',type=str,default="euclidean",
    help='''Pairwise Distance method availabe: braycurtis, canberra, chebyshev,
    cityblock, correlation, cosine, dice, euclidean, hamming, jaccard,
    jensenshannon, kulsinski, mahalanobis, matching, minkowski, rogerstanimoto, 
    russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule''')
    parser.add_argument('-ml',type=str,default="weighted",
    help="Linkage method available: single, complete, average, weighted, centeroid, median, ward")
    parser.add_argument('-nc' ,type=int,default=3,help="Number of Clusters")
    parser.add_argument('-cmt',type=int,default=0,help='''Select Clustering Method: 
    Cluster Hierarchical: 0
    K-Means: 1
    Fuzzy C-Means: 2''')

    args = parser.parse_args()
    return args.d.split(" "),args.c,args.a,args.dt,args.ml,args.nc,args.cmt

[[datasetpath,numSkipedRow,sheetname],cmpMissData,adaptData,distanceMethod,linkageMethod,numCluster,cmt]=readArgs()
strMethod=", "+normStr[adaptData]+", Num clusters= "+str(numCluster)+", distMethod="+distanceMethod+", linkMethod="+linkageMethod

dataset=readExcel(datasetpath,int(numSkipedRow),sheetname)
[labels,data,dataDist,dataLink]=computeExcelData(dataset,cmpMissData,adaptData,distanceMethod,linkageMethod)
clusteringMtd[cmt](data,dataLink,numCluster,strMethod)

plotBiDispersidade(data,None,0,title="Feature 0 with Feature 1"+strMethod)
plotFunction(fancy_dendrogram,dataLink,labels=np.array(dataset.values)[:,0],max_d=(dataLink[-numCluster+1,2]+dataLink[-numCluster,2])/2,title="Denogram, distMethod="+distanceMethod+", linkMethod=" +linkageMethod,
ylabel='Distance',xlabel='ID')


# %%prin

