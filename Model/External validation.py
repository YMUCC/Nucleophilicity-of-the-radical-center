import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import r2_score #R2

# Load training data
data = pd.read_csv("/home/dong/RadicalPolarity/Training/Final_Training/Standardized/xyz_rdkitdescriptors.csv")
names = ['lumo','Charge', 'AN', 'EN', 'Density', 'DP', 'homo', 'BV','ANN','ENN','disp']
x = np.array(data[names]).reshape(-1, len(names))
y = np.array(data['fKN_def2QZVP']).reshape(-1)

# Load new data for predictions
df = pd.read_excel("/home/dong/RadicalPolarity/Training/Final_Training/Standardized/OutofSamples.xlsx", sheet_name='Metal')
y1 = np.array(df['fk']).reshape(-1)
z = np.array(df[names]).reshape(-1, len(names))

# Load the pre-trained model
model = joblib.load('/home/dong/RadicalPolarity/Training/Final_Training/Standardized/RF/def2QZVP/NBO/rf_model.pkl')
model.fit(x, y)

# Make predictions
y1_pred = model.predict(z)

pred_mae = mean_absolute_error(y1, y1_pred)
pred_rmse = np.sqrt(mean_squared_error(y1, y1_pred))
pred_r2 = r2_score(y1, y1_pred)


# Store predictions in DataFrame and save to a file
predictions_df = pd.DataFrame({'Name': df['Name'], 'y1_pred': y1, 'rf_pred': y1_pred})
predictions_df.to_csv('./PredictionsRF.csv', index=False)
print("Predictions saved to Predictions.csv")


