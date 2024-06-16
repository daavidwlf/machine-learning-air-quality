from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from sklearn.metrics import r2_score

def model(X_train, y_train, X_valid, y_valid):
    ln = Sequential([Dense(1, input_shape=(133,), activation='linear')])
    ln.summary()

    ln.compile(optimizer = tf.keras.optimizers.Adam(1e-1), loss = 'mse')

    ln.fit(X_train, y_train, batch_size = 10000, epochs=12, validation_data=(X_valid, y_valid))

    def print_scores(m, X_train=X_train, X_valid=X_valid):
        preds = m.predict(X_valid, 10000)
        print('Train R2 = ', r2_score(y_train, m.predict(X_train, 10000)), 
            ', Valid R2 = ', r2_score(y_valid, preds), ', Valid MSE = ', 
            m.evaluate(X_valid, y_valid, 10000, 0))
    
    print_scores(ln)