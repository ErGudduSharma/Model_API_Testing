import numpy as np
import pandas as pd

df = pd.read_csv('Social_Network_Ads.csv')
df = df.drop(columns=['User ID', 'Gender'], axis=1)

from sklearn.model_selection import train_test_split
x = df.drop(columns='Purchased', axis=1)
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)


import joblib
joblib.dump(rf, 'rf_model.pkl')
print('model saved as rf_model.pkl')