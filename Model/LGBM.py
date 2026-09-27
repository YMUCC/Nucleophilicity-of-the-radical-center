# coding=utf-8
import datetime
import pandas as pd
import numpy as np
import random
import hyperopt
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import r2_score #R2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold #cv
from hyperopt import  hp,fmin,rand,tpe,partial,STATUS_OK,Trials
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
names = ['lumo','Charge', 'AN', 'EN', 'Density', 'DP', 'homo', 'BV','ANN','ENN','disp']
#names = ['lumo','Charge', 'EV', 'EN', 'Density', 'homo', 'disp',]
x = np.array(data[names]).reshape(-1,len(names))
y = np.array(data['fKN_def2QZVP']).reshape(-1)

scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x.shape)
print(y.shape)

printlog("step2: searching parameters...")

random_state=42
n_iter=200

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,shuffle=True,random_state=random_state)

num_folds=10
kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

def hyperopt_cv(params, random_state=random_state, cv=kf, X=x_train, y=y_train):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'num_leaves': int(params['num_leaves']),
        'learning_rate': params['learning_rate'],
        'verbose': -1,
        }
    
    # we use this params to create a new Regressor
    model = lgb.LGBMRegressor(random_state=random_state, **params, n_jobs=-1)
    
    # and then conduct the cross validation with the same folds as before
    MAE = (-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1).mean())
    return {'loss': MAE, 'status': STATUS_OK}

# possible values of parameters
space = {'n_estimators': hp.uniform('n_estimators', 60, 120),
        'max_depth' : hp.uniform('max_depth', 3, 30 ),
        'num_leaves' : hp.uniform('num_leaves', 10, 80),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
        }

# trials will contain logging information
trials = Trials()
early_stop_fn=no_progress_loss(40)

best = fmin(fn=hyperopt_cv, # function to optimize
        space=space,
        algo=tpe.suggest, # optimization algorithm, hyperotp will select its parameters automatically
        max_evals=n_iter, # maximum number of iterations
        trials=trials, # logging
        early_stop_fn=early_stop_fn, # control early_stop
        rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
        )
for trial in trials:
    print(trial)
print("\n"+"Best params:",best)

printlog("step3: training and evaluting model using best params...")

# building and evaluating final model using best params
model = lgb.LGBMRegressor(random_state=random_state,
            n_estimators=int(best['n_estimators']),
            max_depth=int(best['max_depth']),
            num_leaves=int(best['num_leaves']), 
            learning_rate=best['learning_rate'],
            )
model.fit(x_train,y_train)

y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

MAE = mean_absolute_error(y_test, y_pred_test)
TrMAE = mean_absolute_error(y_train, y_pred_train)
print("MAE: %.3f" % MAE)
print("TrMAE: %.3f" % TrMAE)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
TrRMSE = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("RMSE: %.3f" % RMSE)
print("TrRMSE: %.3f" % TrRMSE)
print("score: %.3f" % r2_score(y_test, y_pred_test))
print("Trscore: %.3f" % r2_score(y_train, y_pred_train))

joblib.dump(model, filename='lgbm_model.pkl')

# plots of params using hyperopt
plt.style.use('bmh') # bmh, classic, ggplot, seaborn-pastel, fivethirtyeight, seaborn-paper
results=np.array([[x['result']['loss'],
                    x['misc']['vals']['n_estimators'][0],
                    x['misc']['vals']['max_depth'][0],
                    x['misc']['vals']['num_leaves'][0],
                    x['misc']['vals']['learning_rate'][0]] for x in trials.trials])

results_df=pd.DataFrame(results,
                        columns=['MAE', 'n_estimators', 'max_depth', 'num_leaves', 'learning_rate'])
results_df.plot(subplots=True,figsize=(10, 10))
plt.xlabel('Trials',fontsize=14,fontweight='bold')
plt.rcParams.update({'font.family':'sans-serif'})
plt.savefig('params-LGBM.png',dpi=400)
plt.close()

# plots of y_pred and y_true for test and train
plt.plot(y_pred_train,y_train,'ro',label='Train')
plt.plot(y_pred_test,y_test,'bs',label='Test')
plt.legend(['Train','Test'])
plt.xlabel('Predicted M-H Distance ('+u'\u00C5'+')',fontsize=14,fontweight='bold')
plt.ylabel('DFT Calculated M-H Distance ('+u'\u00C5'+')',fontsize=14,fontweight='bold')
plt.rcParams.update({'font.family':'sans-serif'})
#plt.xlim(1.38, 2.12)
#plt.ylim(1.38, 2.12)
plt.savefig('LGBM.png',dpi=400)
plt.close()

# Hist of y_pred and y_true for test and train
fig,ax = plt.subplots()
ax.hist((y_pred_train - y_train), bins=20)
ax.hist((y_pred_test - y_test), bins=20)
plt.xlabel('Prediction Error ('+u'\u00C5'+')',fontsize=14,fontweight='bold')
plt.ylabel('Counts',fontsize=14,fontweight='bold')
plt.legend(['Train','Test'])
plt.savefig('Hist-LGBM.png',dpi=400)
plt.close()

# plots of params using hyperopt
parameters = ['n_estimators', 'max_depth', 'num_leaves', 'learning_rate'] # LGBM
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
    axes[i].set_ylabel('MAE ('+u'\u00C5'+')',fontsize=14,fontweight='bold')
    #axes[i].set_title(val)
    #axes[i].set_ylim([0.38,0.48])
plt.savefig('paramsMAE.png',dpi=400)
plt.close()

if __name__ == '__main__':
    p = Process(target=hyperopt_cv, args=('bob',))
    p.start()
    p.join()

t2 = time()
time = t2 - t1
printlog("task end...")
print("wall time (s): %.3f" % time)


