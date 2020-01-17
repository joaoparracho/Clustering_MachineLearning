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
[Inputs,Outputs,dataTrain,dataTest,outputTrain,outputTest]=divideExcelData(dataset,cmpMissData,trainP)
[Y_pred,Y_pred_Test]=regressionMtd[rmt](dataTrain,dataTest,outputTrain,outputTest)

[MAE_regression_Test,MSE_regression_Test,RMSE_regression_Test,Errors_regression_Test,SSE_regression_Test,MAPE_regression_Test]=evaluateErrorMetric(outputTest,Y_pred_Test)
print("EVALUATE ERROR METRICS:",MAE_regression_Test,"",MSE_regression_Test,"",RMSE_regression_Test,"",Errors_regression_Test,"",SSE_regression_Test,"",MAPE_regression_Test)
BOXPLOTAnalysis(outputTrain,Y_pred,Errors_regression_Test)

# Cross Correlation and Auto Correlation Analysis
for x in range(0, 3):
    print(np.corrcoef(Inputs[:,x].astype(float),Outputs.astype(float)))


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(Outputs.astype(float)) # Autocorrelation of Output
plt.show()
