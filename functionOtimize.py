import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import skfuzzy  as  fuzz
import plotly.express as px
import random
from scipy.spatial.distance import pdist,squareform #funções pdist e square form deve ser obtidas a partir do package scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error 
#git reset --hard
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.graphics.tsaplots import plot_acf
import os


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

def fancy_boxplot(*args, **kwargs):
    facecolor = kwargs.pop('facecolor', None)
    labels = kwargs.pop('labels', None)
    boxes=plt.boxplot(*args, **kwargs, patch_artist=True)
    for x in range(0,len(boxes["boxes"])):
        plt.setp(boxes["boxes"][x],color=facecolor[x])
        plt.setp(boxes["fliers"][x], markeredgecolor=facecolor[x])
    plt.legend(boxes["boxes"], labels,loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=int(len(boxes["boxes"])/3))  

def randomColor(sizeRegressionMtd,sizeMode):
    number_of_colors = sizeRegressionMtd
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    colors=np.repeat(color, sizeMode)
    return colors    

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

def plot3Dispersidade(data3D,cluster,centroids=[],title="3Dscatter"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data3D[:,0], data3D[:,1], data3D[:,2], c=cluster,cmap='viridis', marker='o')
    if len(centroids)>0:
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c='black',s=200,alpha=0.5)
    ax.set_xlabel('Feature X')
    ax.set_ylabel('Feature Y')
    ax.set_zlabel('Feature Z')

    plt.show()
    
def readExcel(excelPath,numSkipedRow=0,sheetName="Clustering"):
    return pd.read_excel(excelPath,sheetName,skiprows=numSkipedRow)

def computeExcelData(excelDataSet,cmpMissData=1,adaptData=1,distanceMethod="euclidean",linkageMethod="average"):
    #Normalização é feita por feature 
    #https://scikit-learn.org/stable/modules/preprocessing.html
    excelDataSet.fillna(excelDataSet.mean(),inplace=True) if cmpMissData else  excelDataSet.dropna(inplace=True) 
    data=normalizeMtd[adaptData](np.array(excelDataSet.values)[:,1:])
    dataDist=pdist(data,distanceMethod)
    dataLink=linkage(data,linkageMethod)
    print(excelDataSet.isnull().sum())
    print(squareform(dataDist))
    return excelDataSet.columns.values,data,dataDist,dataLink

def divideExcelData(excelDataSet,cmpMissData=1):
    excelDataSet.fillna(excelDataSet.mean(),inplace=True) if cmpMissData else  excelDataSet.dropna(inplace=True) 
    Inputs=np.array(excelDataSet)[:,1:len(excelDataSet.columns)-1]
    Outputs=np.array(excelDataSet)[:,len(excelDataSet.columns)-1]
    # Cross Correlation and Auto Correlation Analysis
    # Direct Normal Solar (kW) --> 0.53904677
    # Occupancy Factor --> 0.8804244
    # Wind Speed (m/s) --> 0.21149389
    for x in range(0, 3):
        print(np.corrcoef(Inputs[:,x].astype(float),Outputs.astype(float)))
    
    print('\n')  
    # Autocorrelation of Output --> segundo a autocorrelação a "periodo" de repetição é de 169/24=7dias 
    plot_acf(Outputs.astype(float), lags=200)
    plt.savefig("figures/Autocorrelation.png",bbox_inches='tight')  
    plt.close() 

    #1Novembro 7298 --> 31 Dezembro fim dados
    dataTrain=np.array(excelDataSet)[0:7296,1:len(excelDataSet.columns)-1]
    outputTrain=np.array(excelDataSet)[0:7296,len(excelDataSet.columns)-1]
    dataTest=np.array(excelDataSet)[7296:len(excelDataSet),1:len(excelDataSet.columns)-1]
    outputTest= np.array(excelDataSet)[7296:len(excelDataSet),len(excelDataSet.columns)-1]
    # 7days ago
    outTrain7=np.array(outputTrain)[168:len(outputTrain)]
    inOutlessTrain7=np.array(outputTrain)[0:len(outputTrain)-168]
    outTest7=np.array(outputTest)[168:len(outputTest)]
    dataTest7=np.array(outputTest)[0:len(outputTest)-168]
    # 7days ago and best Cross Correlation
    bestCorr=np.array(dataTrain)[0:len(outputTrain)-168,1]
    bestCorrTrain7=np.column_stack((bestCorr,inOutlessTrain7))
    bestCorr=np.array(dataTest)[0:len(dataTest)-168,1]
    bestCorrdataTest7=np.column_stack((bestCorr,dataTest7))
    return Inputs,Outputs,dataTrain,dataTest,outputTrain,outputTest,inOutlessTrain7.reshape(-1,1),bestCorrTrain7,outTrain7,dataTest7.reshape(-1,1),bestCorrdataTest7,outTest7

def treeDscatterData(excelDataSet,cmpMissData=1,adaptData=1,distanceMethod="euclidean",linkageMethod="average"):
    excelDataSet.fillna(excelDataSet.mean(),inplace=True) if cmpMissData else  excelDataSet.dropna(inplace=True) 
    data=(np.array(excelDataSet.values)[:,1:])
    data3D=data
    data3D[:,0]=data[:,0]/data[:,1]
    data3D[:,1]=data[:,2]/data[:,3]
    data3D[:,2]=data[:,4]/data[:,5]
    data3D=normalizeMtd[adaptData](data3D[:,0:3])
    dataDist=pdist(data3D,distanceMethod)
    dataLink=linkage(data3D,linkageMethod)
    return data3D,dataLink

def clusterHAlgorithm(data,dataLink,numCluster,strMethod):
    C=fcluster(dataLink,numCluster,'maxclust')
    centroids=np.zeros([numCluster,len(data[0])])
    numObjCluster=np.zeros([numCluster])
    for i in range(1,numCluster+1):
        centroids[i-1,:]=data[C==i].mean(axis=0)
        numObjCluster[i-1]=(C==i).sum()
    
    writeLog("logs/ClusterHierq"+strMethod+".txt",3,[str(C),str(centroids),str(numObjCluster)],
    ["C","centroids","numObjCluster"])

    plotBar(numObjCluster.astype(int),title='Cluster_Number of objects per Cluster' + strMethod,ylabel='Number of Object',xlabel='Cluster Index',xIndex=len(numObjCluster))
    
    return C,centroids

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
    C=km.predict(data)   

    numObjCluster=np.zeros([C.max()+1])

    for k in range(0,C.max()+1):
        numObjCluster[k]=(km.predict(data)==k).sum()

    writeLog("logs/kMeans"+strMethod+".txt",6,[str(C),str(centroids),str(numObjCluster),str(Sum_of_squared_distances[numCluster-2]),str(silhouette_avg),str(silhouette_values)],
    ["C","centroids","numObjCluster","Sum_of_squared_distances","silhouette_avg","silhouette_values"])

    plotBar(numObjCluster.astype(int),title='Kmeans_Number of objects per Cluster'+strMethod, 
    ylabel='Number of Object',xlabel='Cluster Index',xIndex=len(numObjCluster))

    return C,centroids

def fuzzyCmeansAlgorithm(data,dataL,numCluster,strMethod):
    fpcs = []
    centroids, membershipDeg, u0, d, ObjFunction, p, fpc = fuzz.cluster.cmeans(data.T, numCluster, 2, error=0.005,maxiter=1000, init=None)
    cluster_membership = np.argmax(membershipDeg, axis=0)
    fpcs.append(fpc)

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
    
    return C,cluster_membership

#def linearRegressionF(dataTrain,dataTest,outputTrain,outputTest):  
def linearRegressionF(dataTrain,dataTest,outputTrain,outputTest,*_):  
    #LINEAR REGRESSION
    LR_mdl=LinearRegression() 
    LR_mdl.fit (dataTrain,outputTrain) 
    Y_pred_LR=LR_mdl.predict (dataTrain) 
    Y_pred_Test_LR=LR_mdl.predict (dataTest) 
    print (LR_mdl.coef_)
    print (LR_mdl.intercept_)
    return Y_pred_LR,Y_pred_Test_LR


def PolynomialRegressionF(dataTrain,dataTest,outputTrain,outputTest,degree,*_):  
    #Polynomial REGRESSION
    poly_features=PolynomialFeatures(degree)
    Inputs_poly=poly_features.fit_transform(dataTrain)
    Inputs_Test_poly=poly_features.fit_transform(dataTest)
    PR_mdl=LinearRegression()
    PR_mdl.fit(Inputs_poly,outputTrain)
    Y_pred_PR=PR_mdl.predict(Inputs_poly)
    Y_pred_Test_PR=PR_mdl.predict(Inputs_Test_poly)
    #print (PR_mdl.coef_)
    #print (PR_mdl.intercept_)
    return Y_pred_PR,Y_pred_Test_PR  

def ANNRegressionF(dataTrain,dataTest,outputTrain,outputTest,deg,nn,act,val,*_):  
    #ANN REGRESSION
    ANN_mdl=MLPRegressor (hidden_layer_sizes = nn, activation =act, max_iter=1000000, verbose = 'False',tol=1e-10, early_stopping=False, validation_fraction=val)
    ANN_mdl.fit(dataTrain,outputTrain)
    Y_pred_ANN=ANN_mdl.predict(dataTrain)
    Y_pred_Test_ANN=ANN_mdl.predict(dataTest)
    #print (ANN_mdl.coefs_)
    #print (ANN_mdl.intercepts_)
    return Y_pred_ANN,Y_pred_Test_ANN 

def SVMRegressionF(dataTrain,dataTest,outputTrain,outputTest,deg,nn,act,val,c,k,ep,*_):  
    #SVM REGRESSION
    SVR_mdl= SVR (C=c,kernel=k,epsilon=ep)
    SVR_mdl =SVR_mdl.fit(dataTrain,outputTrain)
    Y_pred_SVR=SVR_mdl.predict(dataTrain)
    Y_pred_Test_SVR=SVR_mdl.predict(dataTest)
    indexes_SVR=SVR_mdl.support_
    sv=SVR_mdl.support_vectors_
    #print(SVR_mdl.dual_coef_)
    return Y_pred_SVR,Y_pred_Test_SVR 

def SVMGridSearchRegressionF(dataTrain,dataTest,outputTrain,outputTest,*_):  
    #SVR REGRESSION with GridSearch
    find_parameters=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10]}]
    SVR_mdl=GridSearchCV(SVR(),find_parameters,cv=3)
    SVR_mdl.fit(dataTrain,outputTrain)
    SVR_mdl.best_params_
    Y_pred_SVR=SVR_mdl.predict(dataTrain)
    Y_pred_Test_SVR=SVR_mdl.predict(dataTest)
    return Y_pred_SVR,Y_pred_Test_SVR 

def evaluateErrorMetric(outputTest,Y_pred_Test):
    #EVALUATE ERROR METRICS
    # Mean Absolute Error (MAE) – Erro Absoluto Médio
    MAE_regression_Test=mean_absolute_error(outputTest,Y_pred_Test)
    # Mean Squared Error (MSE) – Erro Quadrático Médio
    MSE_regression_Test=mean_squared_error(outputTest,Y_pred_Test)
    # Root Mean Squared Error (RMSE) – Raíz Quadrada do Erro Quadrático Médio
    RMSE_regression_Test=np.sqrt(mean_squared_error(outputTest,Y_pred_Test))
    # Sum of Squared Errors (SSE) – Soma de Erros Quadráticos
    Errors_regression_Test=np.subtract(outputTest,Y_pred_Test)
    SSE_regression_Test=np.sum(Errors_regression_Test*Errors_regression_Test)
    # Mean Absolute Percentage Error (MAPE) – Erro Percentual Absoluto Médio
    Percentual_Errors_regression=np.divide(np.abs(Errors_regression_Test),outputTest)
    MAPE_regression_Test=np.mean(Percentual_Errors_regression)
    return MAE_regression_Test,MSE_regression_Test,RMSE_regression_Test,Errors_regression_Test,SSE_regression_Test,MAPE_regression_Test


def writeLogs(fileName,numObgjW,listStr,listStrTitle,Title,subTitle):
    if os.path.exists(fileName):
      append_write = 'a' 
    else:
     append_write = 'w'
    file1 = open(fileName,append_write) 
    if append_write =='w':
        file1.write("====="+Title+"=====\n")
    file1.write("====="+subTitle+"=====")
    for i in range(0,numObgjW):
        file1.write("\n====="+listStrTitle[i]+"=====\n")
        file1.write(listStr[i])
    file1.write("\n\n")
    file1.close()

def writeLog(fileName,numObgjW,listStr,listStrTitle):
    file1 = open(fileName,'w') 
    for i in range(0,numObgjW):
        file1.write("\n====="+listStrTitle[i]+"=====\n")
        file1.write(listStr[i])
    file1.close()