from functionOtimize import *
import argparse
import itertools
regressionMtd=[linearRegressionF,PolynomialRegressionF,ANNRegressionF,SVMRegressionF,SVMGridSearchRegressionF]
operationMode=["Linear Regression","Polynomial Regression","ANN Regression","SVM Regression","SVM GridSearch Regression"]
mode=["Mode 1","Mode 2","Mode 3"]

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
    parser.add_argument('-rmt',type=int,default=0,help='''Select Regression Method: 
    Linear: 0
    Polynomial: 1
    ANN: 2
    SVM 3
    SVMGridSearch 4''')
    parser.add_argument('-deg' ,type=int,default=2,help="Polynomial degree")
    parser.add_argument('-nn' ,type=int,default=1,help="Number of neurons hidden_layer ANN")
    parser.add_argument('-activation',type=str,default='identity',help="Activation function ANN")
    parser.add_argument('-validation_fraction', type=int,default=0.2,help="Validation Fraction ANN")
    parser.add_argument('-cSVM',type=int,default=5,help="Regularization parameter SVM")
    parser.add_argument('-kernel',type=str,default='rbf',help="Specifies the kernel type to be used in the algorithm linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed'SVM")
    parser.add_argument('-epsilon',type=float,default=0.005,help="Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.")

    args = parser.parse_args()
    return args.d.split(" "),args.c,args.a,args.rmt,args.deg,args.nn,args.activation,args.validation_fraction,args.cSVM,args.kernel,args.epsilon

def runRun(dataTrain,dataTest,outputTrain,outputTest,rmt,mode):
    print(operationMode[rmt]+"-"+mode)
    [Y_pred,Y_pred_Test]=regressionMtd[rmt](dataTrain,dataTest,outputTrain,outputTest,deg,nn,activation,validation_fraction,c,kernel,epsilon)
    [MAE_regression_Test,MSE_regression_Test,RMSE_regression_Test,Errors_regression_Test,SSE_regression_Test,MAPE_regression_Test]=evaluateErrorMetric(outputTest,Y_pred_Test)
    #print("EVALUATE ERROR METRICS","\nMAE",MAE_regression_Test,"\nMSE:",MSE_regression_Test,"\nRMSE:",RMSE_regression_Test,"\nSSE:",SSE_regression_Test,"\nMAPE:",MAPE_regression_Test,"\n")
    Errors_regression_Train=np.subtract(outputTrain,Y_pred)
    Errors_regression=np.concatenate((Errors_regression_Train,Errors_regression_Test))
    plotFunction(plt.boxplot,Errors_regression,0,'bx-',title=str(operationMode[rmt]+"-"+mode), ylabel='Errors',xlabel='Test')
    writeLogs("ERROR METRICS-"+operationMode[rmt]+".txt",3,[str(MAE_regression_Test),str(MSE_regression_Test),str(RMSE_regression_Test),str(SSE_regression_Test),str(MAPE_regression_Test)],["MAE","MSE","RMSE","SSE","MAPE"],"EVALUATE ERROR METRICS\n"+operationMode[rmt],mode)
    return Errors_regression.reshape(-1,1)
   
lastErrors_regression={}
[[datasetpath,numSkipedRow,sheetname],cmpMissData,adaptData,rmt,deg,nn,activation,validation_fraction,c,kernel,epsilon]=readArgs()
dataset=readExcel(datasetpath,int(numSkipedRow),sheetname)
[Inputs,Outputs,dataTrain,dataTest,outputTrain,outputTest,inOutlessTrain7,bestCorrTrain7,outTrain7,dataTest7,bestCorrdataTest7,outTest7]=divideExcelData(dataset,cmpMissData)

for x in range(0,len(regressionMtd)):
    try:
        os.remove("ERROR METRICS-"+operationMode[x]+".txt")
    except:
        pass          

for x in range(0, len(regressionMtd)): 
    lastErrors_regression[(x*3)]=runRun(dataTrain,dataTest,outputTrain,outputTest,x,"Modo1")
    lastErrors_regression[(x*3)+1]=runRun(bestCorrTrain7,bestCorrdataTest7,outTrain7,outTest7,x,"Modo2")
    lastErrors_regression[(x*3)+2]=runRun(inOutlessTrain7,dataTest7,outTrain7,outTest7,x,"Mode3")

plotFunction(fancy_boxplot,lastErrors_regression.values(),0,'bx-',title="Error Boxplot", ylabel='Errors',xlabel='Test',facecolor=randomColor(len(regressionMtd),len(mode)),labels=list(itertools.product(operationMode, mode)))
print("HEY")


