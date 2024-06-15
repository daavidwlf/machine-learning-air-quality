from sklearn.metrics import mean_squared_error

def metrics(X_val, y_val, X_test, y_test, model):

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mse_overall =  mean_squared_error(y_test, y_pred)
    score = model.score(X_val, y_val)

    print("MSE all values: ", mse)
    print("MSE all values overall: ", mse_overall)
    print("R2: ", score)
    return 