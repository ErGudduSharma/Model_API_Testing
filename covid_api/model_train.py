import numpy as np
import pandas as pd

df = pd.read_csv('covid_toy.csv')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
df['fever'] = imputer.fit_transform(df[['fever']])

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = lb.fit_transform(df[col])

x = df.drop(columns='has_covid', axis=1)
y = df['has_covid']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_pred=rf.predict(x_test)

from sklearn.metrics import accuracy_score
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

import joblib
joblib.dump(rf, 'rf_model.pkl')
print('model saved as rf_model.pkl')
