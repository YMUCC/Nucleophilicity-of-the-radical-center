# coding=utf-8
import datetime
import pandas as pd
import numpy as np
import hyperopt
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import r2_score #R2
from sklearn.model_selection import LeaveOneOut, cross_val_score 
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
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
x = np.array(data[names]).reshape(-1,len(names))
y = np.array(data['fKN_def2QZVP']).reshape(-1)

scaler = StandardScaler()
x = scaler.fit_transform(x)
print("X shape:", x.shape)
print("Y shape:", y.shape)


printlog("step2: searching parameters using Leave-One-Out CV...")

random_state=42
n_iter=150

loo = LeaveOneOut() 

def hyperopt_cv(params, random_state=random_state, cv=loo, X=x, y=y):
    # the function gets a set of variable parameters in "param"
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
    }
    
    # we use this params to create a new Regressor
    model = RandomForestRegressor(random_state=random_state, **params, n_jobs=6)
    
    # conduct the cross validation using LeaveOneOut
    MAE = (-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1).mean())
    return {'loss': MAE, 'status': STATUS_OK}

# possible values of parameters
space = {
    'n_estimators': hp.quniform('n_estimators', 745, 755, 1),
    'max_depth' : hp.quniform('max_depth', 230, 240, 1),
}

# trials will contain logging information
trials = Trials()
early_stop_fn = no_progress_loss(20)

best = fmin(
    fn=hyperopt_cv, # function to optimize
    space=space, 
    algo=tpe.suggest, # optimization algorithm
    max_evals=n_iter, # maximum number of iterations
    trials=trials, # logging
    early_stop_fn=early_stop_fn, # control early_stop
    rstate=np.random.RandomState(random_state) # fixing random state for the reproducibility
)
for trial in trials:
    print(trial)
print("\n"+"Best params:", best)

printlog("step3: evaluating model using Leave-One-Out loop with best params...")

best_n_estimators = int(best['n_estimators'])
best_max_depth = int(best['max_depth'])


y_test_loo = []
y_pred_loo = []


fold = 1
for train_index, test_index in loo.split(x):
    # print(f'---*---*--- Fold {fold} ---*---*---')
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test_fold = y[train_index], y[test_index]
    
   
    loop_model = RandomForestRegressor(
        random_state=random_state,
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        n_jobs=-1
    )
    
    loop_model.fit(X_train, y_train)
    y_pred_fold = loop_model.predict(X_test)
    
   
    y_test_loo.append(y_test_fold[0])
    y_pred_loo.append(y_pred_fold[0])
    fold += 1


y_test_loo = np.array(y_test_loo)
y_pred_loo = np.array(y_pred_loo)

printlog("step4: fitting final model on ALL data for deployment and training metrics...")


final_model = RandomForestRegressor(
    random_state=random_state,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    n_jobs=-1
)
final_model.fit(x, y)

y_pred_all_train = final_model.predict(x)


MAE = mean_absolute_error(y_test_loo, y_pred_loo)
TrMAE = mean_absolute_error(y, y_pred_all_train)
print("Test MAE (LOO): %.3f" % MAE)
print("Train MAE (All): %.3f" % TrMAE)

RMSE = np.sqrt(mean_squared_error(y_test_loo, y_pred_loo))
TrRMSE = np.sqrt(mean_squared_error(y, y_pred_all_train))
print("Test RMSE (LOO): %.3f" % RMSE)
print("Train RMSE (All): %.3f" % TrRMSE)

print("Test R2 (LOO): %.3f" % r2_score(y_test_loo, y_pred_loo))
print("Train R2 (All): %.3f" % r2_score(y, y_pred_all_train))


joblib.dump(final_model, filename='rf_model.pkl')

printlog("step5: generating plots...")

# plots of params using hyperopt
plt.style.use('bmh')
results=np.array([[t['result']['loss'],
                    t['misc']['vals']['n_estimators'][0],
                    t['misc']['vals']['max_depth'][0]] for t in trials.trials])

results_df=pd.DataFrame(results, columns=['MAE', 'n_estimators', 'max_depth'])
results_df.plot(subplots=True,figsize=(10, 10))
plt.xlabel('Trials',fontsize=14,fontweight='bold')
plt.rcParams.update({'font.family':'sans-serif'})
plt.savefig('params-RF.png',dpi=400)
plt.close()

# plots of y_pred and y_true for test and train
plt.plot(y_pred_all_train, y, 'ro', label='Train (All Data)')
plt.plot(y_pred_loo, y_test_loo, 'bs', label='Test (LOO)')
plt.legend(['Train','Test'])
plt.xlabel('Predicted BDE('+u'kcal/mol'+')',fontsize=14,fontweight='bold')
plt.ylabel('DFT Calculated BDE('+u'kcal/mol'+')',fontsize=14,fontweight='bold')
plt.rcParams.update({'font.family':'sans-serif'})
plt.savefig('RF.png',dpi=400)
plt.close()

# Hist of y_pred and y_true for test and train
fig,ax = plt.subplots()
ax.hist((y_pred_all_train - y), bins=20, alpha=0.5, label='Train')
ax.hist((y_pred_loo - y_test_loo), bins=20, alpha=0.5, label='Test')
plt.xlabel('Prediction Error('+u'kcal/mol'+')',fontsize=14,fontweight='bold')
plt.ylabel('Counts',fontsize=14,fontweight='bold')
plt.legend(['Train','Test'])
plt.savefig('Hist-RF.png',dpi=400)
plt.close()

# plots of params using hyperopt
parameters = ['n_estimators', 'max_depth'] # RF
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
wall_time = t2 - t1
printlog("task end...")
print("wall time (s): %.3f" % wall_time)