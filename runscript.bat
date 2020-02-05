@echo off
echo.
setlocal enabledelayedexpansion

set /p DIR= Folder name with configs (Example cfg): 

for /R %%y in (%DIR%\*.txt) do (
	set /p ARGS=<%%y
	echo.!ARGS! | findstr /C:"Clustering" 1>nul
	if errorlevel 1 ( 
		python Regression_GroupF.py !ARGS!
	) ELSE ( 
		python Clustering_GrupoF.py !ARGS!
	)
)
:end