from functionOtimize import *
import argparse
regressionMtd=[linearRegressionF,PolynomialRegressionF,ANNRegressionF,SVMRegressionF,SVMGridSearchRegressionF]
def readArgs():
    parser = argparse.ArgumentParser(description='Optimization and Machine Learning (MEEE-IPL).'
                                    ,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d',type=str,default="Datasets_GroupG.xlsx 1 Clustering",
    help='''Dataset Information: 
    1 - Dataset Path (.xlsx)5
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
    parser.add_argument('-nn' ,type=int,default=3,help="Number of neurons")
    parser.add_argument('-tp' ,type=float,default=0.6,help="Train Percentage (0.01 - 0.99)")
    parser.add_argument('-rmt',type=int,default=0,help='''Select Regression Method: 
    Linear: 0
    Polynomial: 1''')

    args = parser.parse_args()
    return args.d.split(" "),args.c,args.a,args.nn,args.tp,args.rmt


[[datasetpath,numSkipedRow,sheetname],cmpMissData,adaptData,numNeurons,trainP,rmt]=readArgs()
#strMethod=", "+normStr[adaptData]+", Num clusters= "+str(numCluster)+", distMethod="+distanceMethod+", linkMethod="+linkageMethod
dataset=readExcel(datasetpath,int(numSkipedRow),sheetname)
Inputs= pd.DataFrame(dataset, columns= ['Direct Normal Solar (kW)','Occupancy Factor','Wind Speed (m/s)']).astype(float).values
Outputs= pd.DataFrame(dataset, columns= ['P (kW)']).astype(float).values
[dataTrain,dataTest,outputTrain,outputTest]=divideExcelData(dataset,cmpMissData,trainP)
[Y_pred,Y_pred_Test]=regressionMtd[rmt](dataTrain[:,4:7],dataTest[:,4:7],outputTrain,outputTest)

[MAE_regression_Test,MSE_regression_Test,RMSE_regression_Test,Errors_regression_Test,SSE_regression_Test,MAPE_regression_Test]=evaluateErrorMetric(outputTest,Y_pred_Test)
print("EVALUATE ERROR METRICS:",MAE_regression_Test,"",MSE_regression_Test,"",RMSE_regression_Test,"",Errors_regression_Test,"",SSE_regression_Test,"",MAPE_regression_Test)
BOXPLOTAnalysis(outputTrain,Y_pred,Errors_regression_Test)

# Cross Correlation and Auto Correlation Analysis
print(np.corrcoef(Inputs[:,0], Outputs[:,0]))
print(np.corrcoef(Inputs[:,1], Outputs[:,0]))
print(np.corrcoef(Inputs[:,2], Outputs[:,0]))
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(Outputs[:,0]) # Autocorrelation of Output
plt.show()
