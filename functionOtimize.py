import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import skfuzzy  as  fuzz
from scipy.spatial.distance import pdist,squareform #funções pdist e square form deve ser obtidas a partir do package scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.model_selection import GridSearchCV

normalizeMtd=[preprocessing.StandardScaler().fit_transform,preprocessing.MinMaxScaler().fit_transform]

def autolabel(rects,ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plotBar(data,**keyParam):
    fig,ax=plt.subplots(figsize=(20.0, 12.0))
    rects1=ax.bar(np.arange(len(data)),data)
    title=keyParam.pop('title')
    xIndex=keyParam.pop('xIndex')
    plt.xticks(np.arange(xIndex),np.arange(1,xIndex+1))
    plt.xlabel(keyParam.pop('xlabel'))
    plt.ylabel(keyParam.pop('ylabel'))
    plt.title(title)
    autolabel(rects1,ax)
    fig.tight_layout()
    plt.savefig('figures/'+title+'.png',bbox_inches='tight')  
    plt.close()

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    plt.axhline(y=max_d, c='k',linewidth=4,label='Cutoff line')
    plt.legend()
    dendrogram(*args, **kwargs)

def plotFunction(function,*positionalParm,**keyParam):
    plt.figure(figsize=(20.0, 12.0))
    title=keyParam.pop('title')
    plt.xlabel(keyParam.pop('xlabel'))
    plt.ylabel(keyParam.pop('ylabel'))
    plt.title(title)
    function(*positionalParm,**keyParam)
    plt.savefig('figures/'+title+'.png',bbox_inches='tight')  
    plt.close()  

def plotBiDispersidade(data,cluster,i,centroids=[],title="scatter"):
    plt.figure(figsize=(20.0, 12.0))
    plt.scatter(data[:,i],data[:,i+1],c=cluster,cmap='viridis',s=50, label='True  Position')#para  permitir  um  gráfico  de  dispersão  dos dados bidimensionais
    plt.xlabel("Feature "+str(i))
    plt.ylabel("Feature "+str(i+1))
    for label, x, y in zip(range(1,len(data)), data[:, 0], data[:, 1]):  # procedimento para incluir etiquetas junto aos pontos 
        plt.annotate(label, xy=(x, y),textcoords='offset points', ha='right', va='bottom')
    
    plt.grid() # para integrar uma grelha
    if len(centroids)>0:
        plt.scatter(centroids[:,i],centroids[:,i+1],c='black',s=200,alpha=0.5)
    plt.savefig('figures/'+ title +'.png',bbox_inches='tight')  
    plt.close()

def readExcel(excelPath,numSkipedRow=0,sheetName="Clustering"):
    return pd.read_excel(excelPath,sheetName,skiprows=numSkipedRow)

def computeExcelData(excelDataSet,cmpMissData=1,adaptData=1,distanceMethod="euclidean",linkageMethod="average"):
    #Normalização é feita por feature 
    #https://scikit-learn.org/stable/modules/preprocessing.html
    excelDataSet.fillna(excelDataSet.mean(),inplace=True) if cmpMissData else  excelDataSet.dropna(inplace=True) 
    data=normalizeMtd[adaptData](np.array(excelDataSet.values)[:,1:])
    dataDist=pdist(data,distanceMethod)
    dataLink=linkage(dataDist,linkageMethod)
    print(excelDataSet.isnull().sum())
    print(squareform(dataDist))
    return excelDataSet.columns.values,data,dataDist,dataLink

def divideExcelData(excelDataSet,cmpMissData=1,trainP=0.7):
    trainSize=int(len(excelDataSet)*trainP+1)
    excelDataSet.fillna(excelDataSet.mean(),inplace=True) if cmpMissData else  excelDataSet.dropna(inplace=True) 
    dataTrain=np.array(excelDataSet)[0:trainSize,0:len(excelDataSet.columns)-1]
    outputTrain=np.array(excelDataSet)[0:trainSize,len(excelDataSet.columns)-1]
    dataTest=np.array(excelDataSet)[trainSize:len(excelDataSet),0:len(excelDataSet.columns)-1]
    outputTest= np.array(excelDataSet)[trainSize:len(excelDataSet),len(excelDataSet.columns)-1]

    return dataTrain,dataTest,outputTrain,outputTest

def clusterHAlgorithm(data,dataLink,numCluster,strMethod):
    C=fcluster(dataLink,numCluster,'maxclust')
    centroids=np.zeros([numCluster,len(data[0])])
    numObjCluster=np.zeros([numCluster])
    for i in range(1,numCluster+1):
        centroids[i-1,:]=data[C==i].mean(axis=0)
        numObjCluster[i-1]=(C==i).sum()
    
    writeLog("logs/ClusterHierq"+strMethod+".txt",3,[str(C),str(centroids),str(numObjCluster)],
    ["C","centroids","numObjCluster"])

    for i in range(0,len(data[0]),2):
        plotBiDispersidade(data,C,i,centroids,title="Cluster_Feature "+str(i)+" with Feature " +str(i+1) + strMethod)

    plotBar(numObjCluster.astype(int),title='Cluster_Number of objects per Cluster' + strMethod, 
    ylabel='Number of Object',xlabel='Cluster Index',xIndex=len(numObjCluster))
    
def kMeansAlgorithm(data,dataLink,numCluster,strMethod):
    
    Sum_of_squared_distances  =  []
    for k in range(2,numCluster+8): # para permitir uma análise para umnúmero variável de clusters
        km=KMeans(n_clusters=k).fit(data)
        Sum_of_squared_distances.append(km.inertia_) #  km.inertia  representaa  soma quadrática das distâncias no interior de cada cluster
        silhouette_avg  =  silhouette_score(data,  km.labels_) # para  obtenção  dos  coeficientes  de Silhouette (media dos valores de Silhouette)
        print("For n_clusters =", k, "The average silhouette_scoreis :", silhouette_avg)
        silhouette_values=silhouette_samples(data,km.labels_) # para   obtenção   dovalor   de Silhouette para cada objeto
        print("For n_clusters =", k, "silhouette_values are :", silhouette_values)
    
    km = KMeans(n_clusters=numCluster).fit(data)#o comando kmeans permite o recurso ao método de clustering K-means, de forma a agrupar as observações identificadas por cada linha de A em N clusters (neste caso 3 clusters foram definidos). 
    silhouette_avg=silhouette_score(data,km.labels_) # para  obtenção  dos  coeficientes  de Silhouette (media dos valores de Silhouette)
    silhouette_values=silhouette_samples(data,km.labels_) # para   obtenção   dovalor   de Silhouette para cada objeto
    centroids=km.cluster_centers_

    plotFunction(plt.plot,range(2,numCluster+8),Sum_of_squared_distances, 'bx-',title='Elbow Method For Optimal k '+strMethod, ylabel='Sum_of_squared_distances',xlabel='k')
    
    for i in range(0,len(data[0]),2):
        plotBiDispersidade(data, km.predict(data),i,centroids,"Kmeans_Feature "+str(i)+" with Feature " +str(i+1)+strMethod)
           
    numObjCluster=np.zeros([km.predict(data).max()+1])

    for k in range(0,km.predict(data).max()+1):
        numObjCluster[k]=(km.predict(data)==k).sum()

    writeLog("logs/kMeans"+strMethod+".txt",6,[str(km.predict(data)),str(centroids),str(numObjCluster),str(Sum_of_squared_distances[numCluster-2]),str(silhouette_avg),str(silhouette_values)],
    ["C","centroids","numObjCluster","Sum_of_squared_distances","silhouette_avg","silhouette_values"])

    plotBar(numObjCluster.astype(int),title='Kmeans_Number of objects per Cluster'+strMethod, 
    ylabel='Number of Object',xlabel='Cluster Index',xIndex=len(numObjCluster))

def fuzzyCmeansAlgorithm(data,dataL,numCluster,strMethod):
    fpcs = []
    centroids, membershipDeg, u0, d, ObjFunction, p, fpc = fuzz.cluster.cmeans(data.T, numCluster, 2, error=0.005,maxiter=1000, init=None)
    cluster_membership = np.argmax(membershipDeg, axis=0)
    fpcs.append(fpc)
    for i in range(0,len(data[0]),2):
        plotBiDispersidade(data, cluster_membership,i,centroids,"FuzzyCmeans_Feature "+str(i)+" with Feature " +str(i+1)+strMethod+", FPC="+str(fpcs))
    
    numObjCluster=np.zeros([numCluster])
    for k in range(0,numCluster):
        numObjCluster[k]=(cluster_membership==k).sum()

    writeLog("logs/FuzzyCMeans"+strMethod+".txt",9,[str(cluster_membership),str(centroids),str(numObjCluster),str(membershipDeg),str(u0),str(d), str(ObjFunction), str(p), str(fpc)],
    ["C","centroids","numObjCluster","membership Degree",
    "u0 - grade of membership of each instance in each considered cluster, the initial state of membership grade",
    "d - before the iterative process, the final Euclidean distance",
    "ObjFunction - the objective function that is iteratively minimized to find the best location for the clusters",
    "p - the totalnumber of iterations run ",
    "fpc - the final fuzzy partition coefficient"])
    
    plotBar(numObjCluster.astype(int),title='FuzzyCmeans_Number of objects per CLuster'+strMethod, 
    ylabel='Number of Object',xlabel='Cluster Index',xIndex=len(numObjCluster))
    
def linearRegressionF(dataTrain,dataTest,outputTrain,outputTest):  
    #LINEAR REGRESSION
    print ('Linear Regression')
    LR_mdl=LinearRegression() 
    LR_mdl.fit (dataTrain,outputTrain) 
    Y_pred_LR=LR_mdl.predict (dataTrain) 
    Y_pred_Test_LR=LR_mdl.predict (dataTest) 
    print (LR_mdl.coef_)
    print (LR_mdl.intercept_)
    return Y_pred_LR,Y_pred_Test_LR


def PolynomialRegressionF(dataTrain,dataTest,outputTrain,outputTest):  
    #Polynomial REGRESSION
    print ('Polynomial Regression')
    poly_features=PolynomialFeatures(degree=2)
    Inputs_poly=poly_features.fit_transform(dataTrain)
    Inputs_Test_poly=poly_features.fit_transform(dataTest)

    PR_mdl=LinearRegression()
    PR_mdl.fit(Inputs_poly,outputTrain)
    Y_pred_PR=PR_mdl.predict(Inputs_poly)
    Y_pred_Test_PR=PR_mdl.predict(Inputs_Test_poly)
    print (PR_mdl.coef_)
    print (PR_mdl.intercept_)
    return Y_pred_PR,Y_pred_Test_PR  

def ANNRegressionF(dataTrain,dataTest,outputTrain,outputTest):  
    #ANN REGRESSION
    print ('ANN Regression')
    ANN_mdl=MLPRegressor (hidden_layer_sizes = 1, activation ='identity', max_iter=1000000, verbose = 'True',tol=1e-10, early_stopping=False, validation_fraction=0.2)
    ANN_mdl.fit(dataTrain,outputTrain)
    Y_pred_ANN=ANN_mdl.predict(dataTrain)
    Y_pred_Test_ANN=ANN_mdl.predict(dataTest)
    print (ANN_mdl.coefs_)
    print (ANN_mdl.intercepts_)
    return Y_pred_ANN,Y_pred_Test_ANN 

def SVMRegressionF(dataTrain,dataTest,outputTrain,outputTest):  
    #SVM REGRESSION
    print ('SVM Regression')
    SVR_mdl= SVR (C=5,kernel='linear',epsilon=0.005)
    SVR_mdl =SVR_mdl.fit(dataTrain,outputTrain)
    Y_pred_SVR=SVR_mdl.predict(dataTrain)
    Y_pred_Test_SVR=SVR_mdl.predict(dataTest)
    indexes_SVR=SVR_mdl.support_
    sv=SVR_mdl.support_vectors_
    print(SVR_mdl.dual_coef_)
    return Y_pred_SVR,Y_pred_Test_SVR 

def SVMGridSearchRegressionF(dataTrain,dataTest,outputTrain,outputTest):  
    #SVR REGRESSION with GridSearch
    find_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000] , 'epsilon': [0.01, 0.05, 0.1, 0.5]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000] , 'epsilon': [0.01, 0.05, 0.1, 0.5]} ]
    SVR_mdl=GridSearchCV(SVR(),find_parameters,cv=3)
    SVR_mdl.fit(dataTrain,outputTrain)
    SVR_mdl.best_params_
    Y_pred_SVR=SVR_mdl.predict(dataTrain)
    Y_pred_Test_SVR=SVR_mdl.predict(dataTest)
    return Y_pred_SVR,Y_pred_Test_SVR 

def evaluateErrorMetric(outputTest,Y_pred_Test):
    #EVALUATE ERROR METRICS
    # MAE Calculus
    MAE_regression_Test=mean_absolute_error(outputTest,Y_pred_Test)
    # MSE Calculus
    MSE_regression_Test=mean_squared_error(outputTest,Y_pred_Test)
    # RMSE Calculus
    RMSE_regression_Test=np.sqrt(mean_squared_error(outputTest,Y_pred_Test))
    # SSE Calculus
    Errors_regression_Test=np.subtract(outputTest,Y_pred_Test)
    SSE_regression_Test=np.sum(Errors_regression_Test*Errors_regression_Test)
    # MAPE Calculus
    Percentual_Errors_regression=np.divide(np.abs(Errors_regression_Test),outputTest)
    MAPE_regression_Test=np.mean(Percentual_Errors_regression)
    return MAE_regression_Test,MSE_regression_Test,RMSE_regression_Test,Errors_regression_Test,SSE_regression_Test,MAPE_regression_Test

def BOXPLOTAnalysis(outputTrain,Y_pred_regression,Errors_regression_Test):  
# BOXPLOT Analysis
    Errors_regression_Train=np.subtract(outputTrain,Y_pred_regression)
    Errors_regression=np.concatenate((Errors_regression_Train,Errors_regression_Test))
    fig, axs = plt.subplots()
    axs.boxplot(Errors_regression)
    axs.set_title('basic plot')
  



def writeLog(fileName,numObgjW,listStr,listStrTitle):
    file1 = open(fileName,"w") 
    for i in range(0,numObgjW):
        file1.write("\n====="+listStrTitle[i]+"=====\n")
        file1.write(listStr[i])
    file1.close()