import pickle
import pandas as pd
import numpy as np


df = pd.read_csv('./opening_prices_biotech_complete.csv')
df = df.drop(['ticker', 'Unnamed: 0'], axis = 1)

new = pd.DataFrame()
new["std"] = df.std(axis=1)
new["min"] = df.min(axis=1)
new["max"] = df.max(axis=1)
new["mean"] = df.mean(axis=1)
new = new.reset_index()

prices = df[df.columns[100:149]]
new = pd.concat([new, prices], axis=1)

new.drop(['index'], axis=1, inplace=True)
new.fillna(0, inplace=True)

model = pickle.load(open('model.sav', 'rb'))
preds = model.predict(new)
true = df[df.columns[149]]
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(true, preds))