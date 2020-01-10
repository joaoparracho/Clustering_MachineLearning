from functionOtimize import *
import argparse

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
    parser.add_argument('-nn' ,type=int,default=3,help="Number of neurons")
    parser.add_argument('-tp' ,type=float,default=0.6,help="Train Percentage (0.01 - 0.99)")
    parser.add_argument('-cmt',type=int,default=0,help='''Select Clustering Method: 
    KNeighbors: 0
    MLP: 1''')

    args = parser.parse_args()
    return args.d.split(" "),args.c,args.a,args.nn,args.tp,args.cmt


[[datasetpath,numSkipedRow,sheetname],cmpMissData,adaptData,numNeurons,trainP,cmt]=readArgs()
#strMethod=", "+normStr[adaptData]+", Num clusters= "+str(numCluster)+", distMethod="+distanceMethod+", linkMethod="+linkageMethod
dataset=readExcel(datasetpath,int(numSkipedRow),sheetname)
[dataTrain,dataTest]=divideExcelData(dataset,cmpMissData,trainP)
print("Here fir ")