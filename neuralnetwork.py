from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
import matplotlib.pyplot as plt
import evaluate

def model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    nn = Sequential([
        Input(shape=(133,)),
        Dense(20, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1, activation='linear')
    ])

    #nn = Sequential([Dense(1, input_shape=(133,), activation='linear')])

    #nn.summary()

    nn.compile(optimizer = tf.keras.optimizers.Adam(0.0001), loss = 'mse')
    print("fitting...")
    history = nn.fit(X_train, y_train, batch_size = 32, epochs=250, validation_data=(X_valid, y_valid), verbose=0)

    # plt.plot(history.history['loss'][1:], label='Training loss')
    # plt.plot(history.history['val_loss'][1:], label='Validation loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    
    evaluate.evaluateMSE(nn, X_train, y_train, X_valid, y_valid, X_test, y_test)
    evaluate.calcR2(nn, X_train, y_train, X_valid, y_valid, X_test, y_test, True)