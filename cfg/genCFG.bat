@echo off
echo.
setlocal enabledelayedexpansion

set datasetpath=Datasets_GroupG.xlsx
set datasheetname[0]=Clustering
set datasheetname[1]=Regression

set mssDt[0]=NCmpMssD
set mssDt[1]=CmpMssD

set adpDt[0]=Normalize
set adpDt[1]=Standardize

set dist[0]=braycurtis
set dist[1]=cosine
set dist[2]=hamming
set dist[3]=euclidean
set dist[4]=chebyshev

REM set dist[1]=canberra
REM set dist[2]=chebyshev
REM set dist[3]=cityblock
REM set dist[4]=correlation
REM set dist[5]=cosine
REM set dist[6]=dice
REM set dist[7]=euclidean
REM set dist[8]=hamming
REM set dist[9]=jaccard
REM set dist[10]=jensenshannon
REM set dist[11]=kulsinski
REM set dist[12]=mahalanobis
REM set dist[13]=matching
REM set dist[14]=minkowski
REM set dist[15]=rogerstanimoto
REM set dist[16]=russellrao
REM set dist[17]=seuclidean
REM set dist[18]=sokalmichener
REM set dist[19]=sokalsneath
REM set dist[20]=sqeuclidean
REM set dist[21]=yule

set lnk[0]=single
set lnk[1]=complete
set lnk[2]=average
set lnk[3]=weighted
set lnk[4]=median
set lnk[5]=ward

set clusterMethod[0]=ClusterHier
set clusterMethod[1]=K-means


for /l %%d in (0, 1, 0) do (
	for /l %%c in (0, 1, 1) do (
		for /l %%a in (0, 1, 1) do (
			for /l %%t in (0, 1, 4) do (
				for /l %%m in (0, 1, 2) do (
					for /l %%n in (2, 1, 6) do (
						for /l %%z in (0, 1, 1) do (
							if not exist OAA_!datasheetname[%%d]!_!mssDt[%%c]!_!adpDt[%%a]!_!dist[%%t]!_NumberCluter-%%n_!clusterMethod[%%z]!.cfg (
								>OAA_!datasheetname[%%d]!_!mssDt[%%c]!_!adpDt[%%a]!_!dist[%%t]!_!lnk[%%m]!_NumberCluter-%%n_!clusterMethod[%%z]!.txt 2>&1(
									echo -d !datasetpath! 1 !datasheetname[%%d]! -c %%c -a %%a -dt !dist[%%t]! -ml !lnk[%%m]! -nc %%n -cmt %%z
								)
							)
						)
					)
				)
			)
		)
	)
)