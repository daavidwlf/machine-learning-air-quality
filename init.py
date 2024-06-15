import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotOnMap
import preprocessing
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import processing

data = pd.read_csv('./AQbench_dataset.csv')

# print(data.isnull().sum())

#preprocess data
X_train, X_val, X_test, y_train, y_val, y_test = processing.process(data)


# plotOnMap.plot(data)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

y_predictions = linear_regression.predict(X_test)

mse = mean_squared_error(y_test, y_predictions, multioutput='raw_values')
mse_overall =  mean_squared_error(y_test, y_predictions)

print(linear_regression.score(X_val, y_val))
print(mse)
print(mse_overall)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(15, input_shape=(139,), activation='linear')
# ])

# model.compile(optimizer='adam', loss='mean_squared_error')

# trained = model.fit(train_data_input, train_data_output, epochs=100, validation_split=0.2)
# loss = model.evaluate(val_data_input, val_data_output)
# print("Test loss:", loss)

# test_pred = model.predict(test_data_input)



# print(pd.get_dummies(val_data['country'], columns=['country']).astype(int))




