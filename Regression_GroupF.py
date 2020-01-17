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
    parser.add_argument('-tp' ,type=float,default=0.6,help="Train Percentage (0.01 - 0.99)")
    parser.add_argument('-rmt',type=int,default=0,help='''Select Regression Method: 
    Linear: 0
    Polynomial: 1
    ANN: 2
    SVM 3
    SVMGridSearch 4''')
    parser.add_argument('-deg' ,type=int,default=2,help="Polynomial degree ")
    parser.add_argument('-nn' ,type=int,default=1,help="Number of neurons hidden_layer")


    args = parser.parse_args()
    return args.d.split(" "),args.c,args.a,args.tp,args.rmt,args.deg,args.nn


[[datasetpath,numSkipedRow,sheetname],cmpMissData,adaptData,trainP,rmt,polynomialDegree,numNeurons]=readArgs()
#strMethod=", "+normStr[adaptData]+", Num clusters= "+str(numCluster)+", distMethod="+distanceMethod+", linkMethod="+linkageMethod
dataset=readExcel(datasetpath,int(numSkipedRow),sheetname)
[Inputs,Outputs,dataTrain,dataTest,outputTrain,outputTest]=divideExcelData(dataset,cmpMissData,trainP)
[Y_pred,Y_pred_Test]=regressionMtd[rmt](dataTrain,dataTest,outputTrain,outputTest)

[MAE_regression_Test,MSE_regression_Test,RMSE_regression_Test,Errors_regression_Test,SSE_regression_Test,MAPE_regression_Test]=evaluateErrorMetric(outputTest,Y_pred_Test)
print("EVALUATE ERROR METRICS","\nMAE",MAE_regression_Test,"\nMSE:",MSE_regression_Test,"\nRMSE:",RMSE_regression_Test,"\nSSE:",SSE_regression_Test,"\nMAPE:",MAPE_regression_Test,"\n")
BOXPLOTAnalysis(outputTrain,Y_pred,Errors_regression_Test)


# Cross Correlation and Auto Correlation Analysis
for x in range(0, 3):
    print(np.corrcoef(Inputs[:,x].astype(float),Outputs.astype(float)))


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(Outputs.astype(float)) # Autocorrelation of Output
plt.show(block=False)
plt.show()


