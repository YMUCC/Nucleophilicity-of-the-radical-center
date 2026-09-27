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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import r2_score #R2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_val_score #cv
from hyperopt import hp,fmin,rand,tpe,partial,STATUS_OK,Trials
from hyperopt.early_stop import no_progress_loss
from multiprocessing import Pool,Process
from sklearn.preprocessing import StandardScaler
from time import time
import joblib
import warnings   # `do not disturbe` mode
warnings.filterwarnings('ignore')

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(info+'...\n')

printlog("step1: reading data...")

t1 = time()
data = pd.read_csv("/home/dong/RadicalPolarity/Training/Final_Training/Standardized/xyz_rdkitdescriptors.csv")

group_file_path = r"/home/dong/RadicalPolarity/Training/Final_Training/Standardized/Group.xlsx"
group_df = pd.read_excel(group_file_path)

if len(data) != len(group_df):
    raise ValueError(f"Data length mismatch! Features: {len(data)}, Groups: {len(group_df)}")

names = ['lumo','Charge', 'AN', 'EN', 'Density', 'DP', 'homo', 'BV','ANN','ENN','disp']
x = np.array(data[names]).reshape(-1,len(names))
y = np.array(data['fKN_def2QZVP']).reshape(-1)

groups = np.array(group_df['group_id']).reshape(-1)

scaler = StandardScaler()
x = scaler.fit_transform(x)
print("X shape:", x.shape)
print("Y shape:", y.shape)
print("Groups shape:", groups.shape)


printlog("step2: searching parameters...")

random_state=42
n_iter=150

gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
train_index, test_index = next(gss_outer.split(x, y, groups))

x_train, x_test = x[train_index], x[test_index]
y_train, y_test = y[train_index], y[test_index]
groups_train = groups[train_index]
groups_test = groups[test_index]

num_folds=10
gss_inner = GroupShuffleSplit(n_splits=num_folds, test_size=0.1, random_state=random_state)

def hyperopt_cv(params, random_state=random_state, cv=gss_inner, X=x_train, y=y_train, g=groups_train):
    params = {'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        }
    
    model = RandomForestRegressor(random_state=random_state, **params, n_jobs=6)
    
    MAE = (-cross_val_score(model, X, y, groups=g, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1).mean())
    return {'loss': MAE, 'status': STATUS_OK}

space = {'n_estimators': hp.quniform('n_estimators', 745, 755, 1),
        'max_depth' : hp.quniform('max_depth', 230, 240, 1),
        }

trials = Trials()
early_stop_fn=no_progress_loss(20)

best = fmin(fn=hyperopt_cv,
        space=space, 
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=trials,
        early_stop_fn=early_stop_fn,
        rstate=np.random.RandomState(random_state)
        )
for trial in trials:
    print(trial)
print("\n"+"Best params:",best)

printlog("step3: training and evaluting model using best params...")

critn = {0:'squared_error', 1:'absolute_error', 2:'poisson'}
model = RandomForestRegressor(random_state=random_state,
            n_estimators=int(best['n_estimators']),
            max_depth=int(best['max_depth']),
            n_jobs=-1,
            )
model.fit(x_train,y_train)

y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

MAE = mean_absolute_error(y_test, y_pred_test)
TrMAE = mean_absolute_error(y_train, y_pred_train)
print("MAE: %.3f" % MAE)
print("Trmae: %.3f" % TrMAE)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
TrRMSE = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("RMSE: %.3f" % RMSE)
print("Trrmse: %.3f" % TrRMSE)
print("R2: %.3f" % r2_score(y_test, y_pred_test))
print("Trscore: %.3f" % r2_score(y_train, y_pred_train))

joblib.dump(model, filename='rf_model.pkl')

plt.style.use('bmh')
results=np.array([[x['result']['loss'],
                    x['misc']['vals']['n_estimators'][0],
                    x['misc']['vals']['max_depth'][0]] for x in trials.trials])

results_df=pd.DataFrame(results,
                        columns=['MAE', 'n_estimators', 'max_depth'])
results_df.plot(subplots=True,figsize=(10, 10))
plt.xlabel('Trials',fontsize=14,fontweight='bold')
plt.rcParams.update({'font.family':'sans-serif'})
plt.savefig('params-RF.png',dpi=400)
plt.close()

plt.plot(y_pred_train,y_train,'ro',label='Train')
plt.plot(y_pred_test,y_test,'bs',label='Test')
plt.legend(['Train','Test'])
plt.xlabel('Predicted BDE('+u'kcal/mol'+')',fontsize=14,fontweight='bold')
plt.ylabel('DFT Calculated BDE('+u'kcal/mol'+')',fontsize=14,fontweight='bold')
plt.rcParams.update({'font.family':'sans-serif'})
plt.savefig('RF.png',dpi=400)
plt.close()

fig,ax = plt.subplots()
ax.hist((y_pred_train - y_train), bins=20)
ax.hist((y_pred_test - y_test), bins=20)
plt.xlabel('Prediction Error('+u'kcal/mol'+')',fontsize=14,fontweight='bold')
plt.ylabel('Counts',fontsize=14,fontweight='bold')
plt.legend(['Train','Test'])
plt.savefig('Hist-RF.png',dpi=400)
plt.close()

parameters = ['n_estimators', 'max_depth']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
cmap = plt.cm.hsv_r
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    ys = np.array(ys)
    axes[i].scatter(xs, ys, s=200, linewidth=0.5, alpha=0.5, c=cmap(float(i)/len(parameters)))
    axes[i].set_xlabel(val,fontsize=14,fontweight='bold')
    axes[i].set_ylabel('MAE (kcal/mol)',fontsize=14,fontweight='bold')
plt.savefig('paramsMAE.png',dpi=400)
plt.close()

t2 = time()
time_cost = t2 - t1
printlog("task end...")
print("wall time (s): %.3f" % time_cost)