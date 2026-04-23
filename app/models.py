import joblib
from sklearn.linear_model import LogisticRegression

def logisticRegressionModel(x_train, y_train):
    model = LogisticRegression(max_iter=5000, class_weight='balanced')
    model.fit(x_train, y_train)

    # Save model to file
    joblib.dump(model, '../artifacts/model.pkl')
    return model