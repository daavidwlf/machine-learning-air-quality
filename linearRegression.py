from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import evaluate

def model(X_train, y_train, X_val, y_val, X_test, y_test, all_targets):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    evaluate.calcMSE(lr, X_train, y_train, X_val, y_val, X_test, y_test, all_targets)
    evaluate.calcR2(lr, X_train, y_train, X_val, y_val, X_test, y_test, False)