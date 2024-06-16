from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def model(X, y):
    linear_regression = LinearRegression()
    linear_regression.fit(X, y)
    return linear_regression