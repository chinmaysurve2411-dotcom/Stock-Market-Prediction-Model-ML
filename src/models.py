from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_models(X_train, y_train):
    lr = LinearRegression()
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf
