# coding=utf-8
import datetime
import pandas as pd
import numpy as np
import random
import hyperopt
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import r2_score #R2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold #cv
from hyperopt import  hp,fmin,rand,tpe,partial,STATUS_OK,Trials
from hyperopt.early_stop import no_progress_loss
from multiprocessing import Pool,Process
from time import time
import joblib
from sklearn.preprocessing import StandardScaler
import warnings   # `do not disturbe` mode
import json  

warnings.filterwarnings('ignore')

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "="*60 + "%s" % nowtime)  
    print(info + '...\n')

printlog("step1: reading data...")

t1 = time()
data = pd.read_csv("/home/dong/RadicalPolarity/Training/Final_Training/Standardized/xyz_rdkitdescriptors.csv")
names = ['lumo','Charge', 'AN', 'EN', 'Density', 'DP', 'homo', 'BV','ANN','ENN','disp']
#names = ['lumo','Charge', 'EV', 'EN', 'Density', 'homo', 'disp',]
x = np.array(data[names]).reshape(-1, len(names))
y = np.array(data['fKN_def2QZVP']).reshape(-1)

scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x.shape)
print(y.shape)

printlog("step2: searching parameters...")

random_state = 42
n_iter = 200

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=random_state)

num_folds = 10
kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)


def hyperopt_cv(params, random_state=random_state, cv=kf, X=x_train, y=y_train):
    if isinstance(params, str):
        try:
            params = json.loads(params)
            print("Converted params to dict")
        except json.JSONDecodeError:
            print(f"Failed to parse: {params}")
            return {'loss': float('inf'), 'status': STATUS_OK}
    
    if not isinstance(params, dict):
        print(f"Params must be dict, got {type(params)}")
        return {'loss': float('inf'), 'status': STATUS_OK}
    
   
    params = {
        'max_depth': int(params['max_depth']), 
        'min_samples_leaf': int(params['min_samples_leaf']),
        'max_features': int(params['max_features']),
    }
    
    model = DecisionTreeRegressor(random_state=random_state, **params)
    MAE = (-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1).mean())
    return {'loss': MAE, 'status': STATUS_OK}

space = {
    'max_depth' : hp.quniform('max_depth', 100, 800, 10),
    'min_samples_leaf' : hp.uniform('min_samples_leaf', 1, 20),
    'max_features' : hp.quniform('max_features', 100, 1000, 10),
}

trials = Trials()
early_stop_fn = no_progress_loss(40)

best = fmin(
    fn=hyperopt_cv,
    space=space, 
    algo=tpe.suggest,
    max_evals=n_iter,
    trials=trials,
    early_stop_fn=early_stop_fn,
    rstate=np.random.RandomState(random_state)
)


print("\nBest params:", str(best))

printlog("step3: training and evaluting model using best params...")

model = DecisionTreeRegressor(
    random_state=random_state, 
    max_depth=int(best['max_depth']),
    min_samples_leaf=int(best['min_samples_leaf']),
    max_features=int(best['max_features']),
)
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

MAE = mean_absolute_error(y_test, y_pred_test)
TrMAE = mean_absolute_error(y_train, y_pred_train)
print(f"MAE: {MAE:.3f}")
print(f"TrMAE: {TrMAE:.3f}")

RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
TrRMSE = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"RMSE: {RMSE:.3f}")
print(f"TrRMSE: {TrRMSE:.3f}")
print(f"R2 score: {r2_score(y_test, y_pred_test):.3f}")
print(f"R2 Trscore: {r2_score(y_train, y_pred_train):.3f}")

joblib.dump(model, filename='dt_model.pkl')


t2 = time()
printlog("task end...")
print(f"wall time (s): {t2 - t1:.3f}")