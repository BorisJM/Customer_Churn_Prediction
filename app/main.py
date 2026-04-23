from encoding import encodingDatabase
from evaluate import predictions
from feature_importance import featureImportance
from load_data import loadData
from models import logisticRegressionModel
from split import splitData
from target import setTarget

df = loadData('../data.csv')
# Target -> Churn
df = setTarget(df)
# Encoding data
df = encodingDatabase(df)
# Split data
x_train, x_test, y_train, y_test = splitData(df)
# Model
model = logisticRegressionModel(x_train, y_train)
# Evaluations, scores
predictions(x_test, y_test, model)
# Feature importance
featureImportance(model, df.drop(['Churn'], axis=1))