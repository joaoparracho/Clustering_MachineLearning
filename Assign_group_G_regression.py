#%%
from functionOtimize import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset=pd.read_excel("Datasets_GroupG.xlsx","Regression")

#plt.plot(dataset.values[:,0], dataset.values[:,4], label='linear')
#plt.legend()
#plt.show()
#dataset=readExcel("Datasets_GroupG.xlsx",int(1),"Regression")

Inputs= pd.DataFrame(dataset, columns= ['DIA','MES','ANO','HORA','Direct Normal Solar (kW)','Occupancy Factor','Wind Speed (m/s)'])
Outputs= pd.DataFrame(dataset, columns= ['P (kW)'])

percentageData=80 
train_size=int(percentageData/100*len(Inputs))

Inputs_Train=np.array(Inputs[0:train_size])
Inputs_Test=np.array(Inputs[train_size:])
Outputs_Train=np.array(Outputs[0:train_size])
Outputs_Test=np.array(Outputs[train_size:])


#LINEAR REGRESSION
print ('Linear Regression')
LR_mdl=LinearRegression(normalize=True) 
LR_mdl.fit (Inputs_Train,Outputs_Train) 
Y_pred_LR=LR_mdl.predict (Inputs) 
Y_pred_Test_LR=LR_mdl.predict (Inputs_Test) 

print (LR_mdl.coef_)
print (LR_mdl.intercept_)

#Polynomial REGRESSION
print ('Polynomial Regression')
poly_features=PolynomialFeatures(degree=2)
Inputs_poly=poly_features.fit_transform(Inputs_Train)
Inputs_Test_poly=poly_features.fit_transform(Inputs_Test)

PR_mdl=LinearRegression()
PR_mdl.fit(Inputs_poly,Outputs_Train)
Y_pred_PR=PR_mdl.predict(Inputs_poly)
Y_pred_Test_PR=PR_mdl.predict(Inputs_Test_poly)
print (PR_mdl.coef_)
print (PR_mdl.intercept_)

#ANN REGRESSION
print ('ANN Regression')
from sklearn.neural_network import MLPRegressor
ANN_mdl=MLPRegressor (hidden_layer_sizes = 1, activation ='identity', max_iter=1000000, verbose = 'True',tol=1e-10, early_stopping=False, validation_fraction=0.2)
ANN_mdl.fit(Inputs_Train,Outputs_Train)
Y_pred_ANN=ANN_mdl.predict(Inputs_Train)
Y_pred_Test_ANN=ANN_mdl.predict(Inputs_Test)
print (ANN_mdl.coefs_)
print (ANN_mdl.intercepts_)

#SVM REGRESSION
print ('SVM Regression')
from sklearn.svm import SVR
SVR_mdl= SVR (C=5,kernel='linear',epsilon=0.005)
SVR_mdl =SVR_mdl.fit(Inputs_Train,Outputs_Train)
Y_pred_SVR=SVR_mdl.predict(Inputs_Train.T)
Y_pred_Test_SVR=SVR_mdl.predict(Inputs_Test)
indexes_SVR=SVR_mdl.support_
sv=SVR_mdl.support_vectors_
print(SVR_mdl.dual_coef_)

#SVR REGRESSION with GridSearch
#from sklearn.model_selection import GridSearchCV
#find_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000] , 'epsilon': [0.01, 0.05, 0.1, 0.5]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000] , 'epsilon': [0.01, 0.05, 0.1, 0.5]} ]
#SVR_mdl=GridSearchCV(SVR(),find_parameters,cv=3)
#SVR_mdl.fit(Inputs_Train,Outputs_Train)
#SVR_mdl.best_params_
#Y_pred_SVR=SVR_mdl.predict(Inputs_Train)
#Y_pred_Test_SVR=SVR_mdl.predict(Inputs_Test)


# EVALUATE ERROR METRICS (only for Test Subset)
from sklearn.metrics import mean_absolute_error, mean_squared_error 
# MAE Calculus
MAE_LR_Test=mean_absolute_error(Outputs_Test,Y_pred_Test_LR)
MAE_PR_Test=mean_absolute_error(Outputs_Test,Y_pred_Test_PR)
MAE_ANN_Test=mean_absolute_error(Outputs_Test,Y_pred_Test_ANN)
MAE_SVR_Test=mean_absolute_error(Outputs_Test,Y_pred_Test_SVR)


# MSE Calculus
MSE_LR_Test=mean_squared_error(Outputs_Test,Y_pred_Test_LR)
MSE_PR_Test=mean_squared_error(Outputs_Test,Y_pred_Test_PR)
MSE_ANN_Test=mean_squared_error(Outputs_Test,Y_pred_Test_ANN)
MSE_SVR_Test=mean_squared_error(Outputs_Test,Y_pred_Test_SVR)

# RMSE Calculus
RMSE_LR_Test=np.sqrt(mean_squared_error(Outputs_Test,Y_pred_Test_LR))
RMSE_PR_Test=np.sqrt(mean_squared_error(Outputs_Test,Y_pred_Test_PR))
RMSE_ANN_Test=np.sqrt(mean_squared_error(Outputs_Test,Y_pred_Test_ANN))
RMSE_SVR_Test=np.sqrt(mean_squared_error(Outputs_Test,Y_pred_Test_SVR))


# SSE Calculus
Errors_LR_Test=np.subtract(Outputs_Test,Y_pred_Test_LR)
Errors_PR_Test=np.subtract(Outputs_Test,Y_pred_Test_PR)
Errors_ANN_Test=np.subtract(Outputs_Test,Y_pred_Test_ANN)
Errors_SVR_Test=np.subtract(Outputs_Test,Y_pred_Test_SVR)

SSE_LR_Test=np.sum(Errors_LR_Test*Errors_LR_Test)
SSE_PR_Test=np.sum(Errors_PR_Test*Errors_PR_Test)
SSE_ANN_Test=np.sum(Errors_ANN_Test*Errors_ANN_Test)
SSE_SVR_Test=np.sum(Errors_SVR_Test*Errors_SVR_Test)

# MAPE Calculus
Percentual_Errors_LR=np.divide(np.abs(Errors_LR_Test),Outputs_Test)
Percentual_Errors_PR=np.divide(np.abs(Errors_PR_Test),Outputs_Test)
Percentual_Errors_ANN=np.divide(np.abs(Errors_ANN_Test),Outputs_Test)
Percentual_Errors_SVR=np.divide(np.abs(Errors_SVR_Test),Outputs_Test)

MAPE_LR_Test=np.mean(Percentual_Errors_LR)
MAPE_PR_Test=np.mean(Percentual_Errors_PR)
MAPE_ANN_Test=np.mean(Percentual_Errors_ANN)
MAPE_SVR_Test=np.mean(Percentual_Errors_SVR)


# BOXPLOT Analysis
import matplotlib.pyplot as plt
Errors_ANN_Train=np.subtract(Outputs_Train,Y_pred_ANN)
Errors_ANN=np.concatenate((Errors_ANN_Train,Errors_ANN_Test))
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(Errors_ANN)
plt.show()


# %%
