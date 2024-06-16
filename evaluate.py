from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def evaluateMSE(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    print('Train MSE = ', model.evaluate(X_train, y_train, verbose=0))
    print('Valid MSE = ', model.evaluate(X_valid, y_valid, verbose=0))
    print('Test MSE = ', model.evaluate(X_test, y_test, verbose=0))

def calcMSE(model, X_train, y_train, X_valid, y_valid, X_test, y_test, all_targets):
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)
    print('Train MSE = ', mean_squared_error(y_train, pred_train))
    print('Valid MSE = ', mean_squared_error(y_valid, pred_valid))
    print('Test MSE = ', mean_squared_error(y_test,  pred_test))
    
    if(all_targets):
        print('Train MSE = ', mean_squared_error(y_train, pred_train, multioutput='raw_values'))
        print('Valdi MSE = ', mean_squared_error(y_valid, y_valid, multioutput='raw_values'))
        print('Test MSE = ', mean_squared_error(y_test,  pred_test, multioutput='raw_values'))
     

def calcR2(model, X_train, y_train, X_valid, y_valid, X_test, y_test, surpress):
    if(surpress):
        pred_train = model.predict(X_train, verbose=0)
        pred_valid = model.predict(X_valid, verbose=0)
        pred_test = model.predict(X_test, verbose=0)
    else:
        pred_train = model.predict(X_train)
        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)
    print('Train R2 = ', r2_score(y_train, pred_train))
    print('Valid R2 = ', r2_score(y_valid, pred_valid))
    print('Test R2 = ',  r2_score(y_test, pred_test))