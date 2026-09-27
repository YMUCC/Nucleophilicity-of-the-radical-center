# coding=utf-8
import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
z = np.array(data['Atom']).reshape(-1)
name = np.array(data['Name']).reshape(-1)
indice = np.array(data['ID']).reshape(-1)

scaler = StandardScaler()
x = scaler.fit_transform(x)

print(x.shape)
print(x)
print(y.shape)
print(y)
print(z)
print(name)
print(x[:,0])

df = pd.read_excel("/home/dong/RadicalPolarity/Training/Final_Training/Standardized/OutofSamples.xlsx", sheet_name='Alls')
x1 = np.array(df[names]).reshape(-1,len(names))
y1 = np.array(df['fk']).reshape(-1)
name1 = np.array(df['Name']).reshape(-1)

scaler = StandardScaler()
x1 = scaler.fit_transform(x1)

printlog("step2: training 11 models using best params...")

random_state=42
#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=random_state)
x_train, x_test, y_train, y_test, name_train, name_test, indice_train, indice_test = train_test_split(x, y, name, indice, test_size=0.1, random_state=42)

# 定义每个算法
# building and evaluating final model using best params
def DT(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/DT/def2QZVP/NBO/dt_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def GPR(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/GPR/def2QZVP/NBO/gpr_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def KNN(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/KNN/def2QZVP/NBO/knn_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def KRR(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/KRR/def2QZVP/NBO/krr_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def Lasso(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/Lasso/def2QZVP/NBO/lasso_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def LGBM(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/LGBM/def2QZVP/NBO/lgbm_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def NN(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/NN/def2QZVP/NBO/nn_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def RF(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/RF/def2QZVP/NBO/rf_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def svr(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/SVR/def2QZVP/NBO/svr_model.pkl')
    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def XGB(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/XGB/def2QZVP/NBO/xgb_model.pkl')
    #model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

def SR(x_train, y_train, x_test):
    model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/XGB/def2QZVP/NBO/xgb_model.pkl')
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y1_pred = model.predict(x1)
    y_trainE = y_train_pred - y_train
    y_testE = y_test_pred - y_test
    y1_predE = y1_pred - y1
    return y_train_pred, y_test_pred, y_trainE, y_testE, y1_pred, y1_predE

# 调用每个算法，将训练和测试预测结果添加到列表中
train_preds = []
test_preds = []
train_errors = []
test_errors = []
pred_y1 = []
pred_errors = []

dt_train_preds, dt_test_preds, dt_train_errors, dt_test_errors, dt_pred_y1, dt_pred_errors = DT(x_train, y_train, x_test)
train_preds.append(dt_train_preds)
test_preds.append(dt_test_preds)
train_errors.append(dt_train_errors)
test_errors.append(dt_test_errors)
pred_y1.append(dt_pred_y1)
pred_errors.append(dt_pred_errors)

gpr_train_preds, gpr_test_preds, gpr_train_errors, gpr_test_errors, gpr_pred_y1, gpr_pred_errors = GPR(x_train, y_train, x_test)
train_preds.append(gpr_train_preds)
test_preds.append(gpr_test_preds)
train_errors.append(gpr_train_errors)
test_errors.append(gpr_test_errors)
pred_y1.append(gpr_pred_y1)
pred_errors.append(gpr_pred_errors)

knn_train_preds, knn_test_preds, knn_train_errors, knn_test_errors, knn_pred_y1, knn_pred_errors = KNN(x_train, y_train, x_test)
train_preds.append(knn_train_preds)
test_preds.append(knn_test_preds)
train_errors.append(knn_train_errors)
test_errors.append(knn_test_errors)
pred_y1.append(knn_pred_y1)
pred_errors.append(knn_pred_errors)

krr_train_preds, krr_test_preds, krr_train_errors, krr_test_errors, krr_pred_y1, krr_pred_errors = KRR(x_train, y_train, x_test)
train_preds.append(krr_train_preds)
test_preds.append(krr_test_preds)
train_errors.append(krr_train_errors)
test_errors.append(krr_test_errors)
pred_y1.append(krr_pred_y1)
pred_errors.append(krr_pred_errors)

lasso_train_preds, lasso_test_preds, lasso_train_errors, lasso_test_errors, lasso_pred_y1, lasso_pred_errors = Lasso(x_train, y_train, x_test)
train_preds.append(lasso_train_preds)
test_preds.append(lasso_test_preds)
train_errors.append(lasso_train_errors)
test_errors.append(lasso_test_errors)
pred_y1.append(lasso_pred_y1)
pred_errors.append(lasso_pred_errors)

lgbm_train_preds, lgbm_test_preds, lgbm_train_errors, lgbm_test_errors, lgbm_pred_y1, lgbm_pred_errors = LGBM(x_train, y_train, x_test)
train_preds.append(lgbm_train_preds)
test_preds.append(lgbm_test_preds)
train_errors.append(lgbm_train_errors)
test_errors.append(lgbm_test_errors)
pred_y1.append(lgbm_pred_y1)
pred_errors.append(lgbm_pred_errors)

nn_train_preds, nn_test_preds, nn_train_errors, nn_test_errors, nn_pred_y1, nn_pred_errors = NN(x_train, y_train, x_test)
train_preds.append(nn_train_preds)
test_preds.append(nn_test_preds)
train_errors.append(nn_train_errors)
test_errors.append(nn_test_errors)
pred_y1.append(nn_pred_y1)
pred_errors.append(nn_pred_errors)

rf_train_preds, rf_test_preds, rf_train_errors, rf_test_errors, rf_pred_y1, rf_pred_errors = RF(x_train, y_train, x_test)
train_preds.append(rf_train_preds)
test_preds.append(rf_test_preds)
train_errors.append(rf_train_errors)
test_errors.append(rf_test_errors)
pred_y1.append(rf_pred_y1)
pred_errors.append(rf_pred_errors)

svr_train_preds, svr_test_preds, svr_train_errors, svr_test_errors, svr_pred_y1, svr_pred_errors = svr(x_train, y_train, x_test)
train_preds.append(svr_train_preds)
test_preds.append(svr_test_preds)
train_errors.append(svr_train_errors)
test_errors.append(svr_test_errors)
pred_y1.append(svr_pred_y1)
pred_errors.append(svr_pred_errors)

xgb_train_preds, xgb_test_preds, xgb_train_errors, xgb_test_errors, xgb_pred_y1, xgb_pred_errors = XGB(x_train, y_train, x_test)
train_preds.append(xgb_train_preds)
test_preds.append(xgb_test_preds)
train_errors.append(xgb_train_errors)
test_errors.append(xgb_test_errors)
pred_y1.append(xgb_pred_y1)
pred_errors.append(xgb_pred_errors)

sr_train_preds, sr_test_preds, sr_train_errors, sr_test_errors, sr_pred_y1, sr_pred_errors = SR(x_train, y_train, x_test)
train_preds.append(sr_train_preds)
test_preds.append(sr_test_preds)
train_errors.append(sr_train_errors)
test_errors.append(sr_test_errors)
pred_y1.append(sr_pred_y1)
pred_errors.append(sr_pred_errors)

# 将预测结果保存到CSV文件中
df_train = pd.DataFrame(train_preds).T
df_train.columns = ["dt_train", "gpr_train", "knn_train", "krr_train", "lasso_train", "lgbm_train", "nn_train", "rf_train", "svr_train", "xgb_train", "sr_train"]
df_train["name_train"] = name_train.tolist()
df_train["id_train"] = indice_train.tolist()
df_train["y_train"] = y_train.tolist()

df_test = pd.DataFrame(test_preds).T
df_test.columns = ["dt_test", "gpr_test", "knn_test", "krr_test", "lasso_test", "lgbm_test", "nn_test", "rf_test", "svr_test", "xgb_test", "sr_test"]
df_test["name_test"] = name_test.tolist()
df_test["id_test"] = indice_test.tolist()
df_test["y_test"] = y_test.tolist()

df_train_error = pd.DataFrame(train_errors).T
df_train_error.columns = ["dt_trainE", "gpr_trainE", "knn_trainE", "krr_trainE", "lasso_trainE", "lgbm_trainE", "nn_trainE", "rf_trainE", "svr_trainE", "xgb_trainE", "sr_trainE"]
df_test_error = pd.DataFrame(test_errors).T
df_test_error.columns = ["dt_testE", "gpr_testE", "knn_testE", "krr_testE", "lasso_testE", "lgbm_testE", "nn_testE", "rf_testE", "svr_testE", "xgb_testE", "sr_testE"]

df_pred = pd.DataFrame(pred_y1).T
df_pred.columns = ["dt_pred", "gpr_pred", "knn_pred", "krr_pred", "lasso_pred", "lgbm_pred", "nn_pred", "rf_pred", "svr_pred", "xgb_pred", "sr_pred"]
df_pred["name_pred"] = name1.tolist()
df_pred["y1"] = y1.tolist()
df_pred_error = pd.DataFrame(pred_errors).T
df_pred_error.columns = ["dt_predE", "gpr_predE", "knn_predE", "krr_predE", "lasso_predE", "lgbm_predE", "nn_predE", "rf_predE", "svr_predE", "xgb_predE", "sr_predE"]

df = pd.concat([df_train, df_test, df_train_error, df_test_error, df_pred, df_pred_error], axis=1)
df.to_csv("Predictionstrain_test_predRP.csv", index=False)

t2 = time()
time = t2 - t1
printlog("task end...")
print("wall time (s): %.2f" % time)


