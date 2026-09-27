# coding=utf-8
import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import r2_score #R2
from sklearn.model_selection import train_test_split
from multiprocessing import Pool,Process
from time import time
import joblib
from sklearn.preprocessing import StandardScaler
import warnings   # `do not disturbe` mode
warnings.filterwarnings('ignore')

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(info+'...\n')

printlog("step1: reading data...")

t1 = time()
data = pd.read_csv("/home/dong/RadicalPolarity/Training/Final_Training/Standardized/xyz_rdkitdescriptors.csv")
names = ['lumo','Charge', 'AN', 'EN', 'Density', 'DP', 'homo', 'BV','ANN','ENN','disp']
x = np.array(data[names]).reshape(-1,len(names))
y = np.array(data['fKN_def2QZVP']).reshape(-1)
#z = np.array(data['Metal']).reshape(-1)
sample_name = np.array(data['ID']).reshape(-1)

scaler = StandardScaler()
x = scaler.fit_transform(x)

print(x.shape)
print(x)
print(y.shape)
print(y)
#print(z)
print(sample_name)
print(x[:,0])

df = pd.read_excel("/home/dong/RadicalPolarity/Training/Final_Training/Standardized/OutofSamples.xlsx", sheet_name='Alls')
x1 = np.array(df[names]).reshape(-1,len(names))
y1 = np.array(df['fk']).reshape(-1)
w1 = np.array(df['Name']).reshape(-1)

scaler = StandardScaler()
x1 = scaler.fit_transform(x1)

printlog("step2: training 11 models using best params...")

random_state=42
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,shuffle=True,random_state=random_state)
#x_train, x_test, y_train, y_test, sample_names_train, sample_names_test = train_test_split(x, y, sample_name, test_size=0.2, random_state=42)


# building and evaluating final model using best params

dt = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/DT/def2QZVP/NBO/dt_model.pkl')
gpr = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/GPR/def2QZVP/NBO/gpr_model.pkl')
knn = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/KNN/def2QZVP/NBO/knn_model.pkl')
krr = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/KRR/def2QZVP/NBO/krr_model.pkl')
lasso = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/Lasso/def2QZVP/NBO/lasso_model.pkl')
lgbm = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/LGBM/def2QZVP/NBO/lgbm_model.pkl')
nn = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/NN/def2QZVP/NBO/nn_model.pkl')
rf = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/RF/def2QZVP/NBO/rf_model.pkl')
svr = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/SVR/def2QZVP/NBO/svr_model.pkl')
xgboost = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/XGB/def2QZVP/NBO/xgb_model.pkl')
sr = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/XGB/def2QZVP/NBO/xgb_model.pkl')

models = [('DT', dt), ('GPR', gpr), ('KNN', knn), ('KRR', krr), ('Lasso', lasso), ('LGBM', lgbm), ('NN', nn), ('RF', rf), ('SVR', svr), ('XGB', xgboost), ('SR', sr)]


results = pd.DataFrame(columns=['Algorithm', 'MAE_train', 'MAE_test', 'RMSE_train', 'RMSE_test', 'R2_train', 'R2_test', 'MAE_pred', 'RMSE_pred', 'R2_pred'])


for name, model in models:
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    pred_mae = mean_absolute_error(y1, y1_pred)
    pred_rmse = np.sqrt(mean_squared_error(y1, y1_pred))
    pred_r2 = r2_score(y1, y1_pred)
    results = results._append({'Algorithm': name, 'MAE_train': train_mae, 'MAE_test': test_mae, 'RMSE_train': train_rmse, 'RMSE_test': test_rmse,
                'R2_train': train_r2, 'R2_test': test_r2, 'MAE_pred': pred_mae, 'RMSE_pred': pred_rmse, 'R2_pred': pred_r2,}, ignore_index=True)


results.to_csv('Evaluate_ResultsRP.csv', index=False)

t2 = time()
time = t2 - t1
printlog("task end...")
print("wall time (s): %.2f" % time)


