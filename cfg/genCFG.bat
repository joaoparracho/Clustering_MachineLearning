@echo off
echo.
setlocal enabledelayedexpansion

set datasetpath=Datasets_GroupG.xlsx
set datasheetname[0]=Clustering
set datasheetname[1]=Regression

set /p mod= Generate Configs for: Clustering (0), Regression (1), Both (2): 

if /i %mod% == 0 goto :clustering
if /i %mod% == 2 goto :clustering
if /i %mod% == 1 goto :regression

:clustering
set mssDt[0]=NCmpMssD
set mssDt[1]=CmpMssD

set adpDt[0]=Normalize
set adpDt[1]=Standardize

set dist[0]=hamming
set dist[1]=cosine
set dist[2]=hamming
set dist[3]=euclidean
set dist[4]=chebyshev
set dist[5]=braycurtis

set lnk[0]=single
set lnk[1]=complete
set lnk[2]=average
set lnk[3]=weighted
set lnk[4]=ward
set lnk[5]=median


set clusterMethod[0]=ClusterHier
set clusterMethod[1]=K-means
set clusterMethod[2]=FuzzyCMeans


for /l %%d in (0, 1, 0) do (
	for /l %%c in (0, 1, 1) do (
		for /l %%a in (0, 1, 1) do (
			for /l %%t in (0, 1, 3) do (
				for /l %%m in (1, 1, 4) do (
					for /l %%n in (2, 1, 4) do (
						for /l %%z in (0, 1, 2) do (
							if not exist OAA_!datasheetname[%%d]!_!mssDt[%%c]!_!adpDt[%%a]!_!dist[%%t]!_NumberCluter-%%n_!clusterMethod[%%z]!.cfg (
								>OAA_!datasheetname[%%d]!_!mssDt[%%c]!_!adpDt[%%a]!_!dist[%%t]!_!lnk[%%m]!_NumberCluter-%%n_!clusterMethod[%%z]!.txt 2>&1(
									echo -d "!datasetpath! 1 !datasheetname[%%d]!" -c %%c -a %%a -dt !dist[%%t]! -ml !lnk[%%m]! -nc %%n -cmt %%z
								)
							)
						)
					)
				)
			)
		)
	)
)
if /i %mod% == 2 goto :regression
goto :end
:regression
set regressionMethod[0]=Linear
set regressionMethod[1]=Polynomial
set regressionMethod[2]=ANN
set regressionMethod[3]=SVM
set regressionMethod[5]=SVMGridSearch


set kernel[0]=rbf
set	kernel[1]=sigmoid
set kernel[2]=precomputed

set activation[0]=identity
set activation[1]=tanh
set activation[2]=relu
set activation[3]=logistic

for /l %%d in (1, 1, 1) do (
	for /l %%b in (1, 1, 6) do (
		for /l %%c in (0, 1, 3) do (
			for /l %%k in (0, 1, 2) do (
				for /l %%a in (1, 1, 3) do (
					>OAA_!datasheetname[%%d]!_NumberNeuron-%%b_deg-%%a_kernel-!kernel[%%k]!_activation-!activation[%%c]!.txt 2>&1(
						echo -d "!datasetpath! 1 !datasheetname[%%d]!" -deg %%a -nn %%b -activation !activation[%%c]! -kernel !kernel[%%k]!
					)
				)
			)
		)
	)
)
goto :end

:end
echo Finish

