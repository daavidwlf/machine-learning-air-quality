import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf

import plotOnMap
import processing
import linearregression
import evaluate
import neuralnetwork

data = pd.read_csv('./AQbench_dataset.csv')

# print(data.isnull().sum())

#preprocess data
X_train, X_val, X_test, y_train, y_val, y_test = processing.process(data)

print("------------ Linear Regression ------------")
model_lr = linearregression.model(X_train, y_train)
evaluate.metrics(X_val, y_val, X_train, y_train, model_lr)
print("------------ Neural Network ------------")
model_nn = neuralnetwork.model(X_train, y_train.iloc[:,0], X_val, y_val.iloc[:,0])


# plotOnMap.plot(data)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(15, input_shape=(139,), activation='linear')
# ])

# model.compile(optimizer='adam', loss='mean_squared_error')

# trained = model.fit(train_data_input, train_data_output, epochs=100, validation_split=0.2)
# loss = model.evaluate(val_data_input, val_data_output)
# print("Test loss:", loss)

# test_pred = model.predict(test_data_input)



# print(pd.get_dummies(val_data['country'], columns=['country']).astype(int))




