from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def model(X_train, y_train, X_valid, y_valid):
    nn = Sequential([
        Input(shape=(133,)),
        Dense(20, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1, activation='relu')
    ])
    nn.summary()

    nn.compile(optimizer = tf.keras.optimizers.Adam(0.0001), loss = 'mse')

    nn.fit(X_train, y_train, batch_size = 32, epochs=250, validation_data=(X_valid, y_valid))

    plt.plot(nn.history.history['loss'][1:], label='Training loss')
    plt.plot(nn.history.history['val_loss'][1:], label='Validation loss')

    def print_scores(m, X_train=X_train, X_valid=X_valid):
        preds = m.predict(X_valid)
        print('Train R2 = ', r2_score(y_train, m.predict(X_train)), 
            ', Valid R2 = ', r2_score(y_valid, preds), ', Valid MSE = ', 
            m.evaluate(X_valid, y_valid))
    
    print_scores(nn)