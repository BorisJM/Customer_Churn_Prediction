import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function for splitting data, train and test
def splitData(df):
    x = df.drop(['Churn'], axis=1)
    y = df['Churn']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Scale data
    scaler = StandardScaler()
    # Save scaler to file
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    joblib.dump(scaler, '../artifacts/scaler.pkl')
    # Save columns to the file
    joblib.dump(df.columns.tolist(), '../artifacts/columns.pkl')
    return x_train, x_test, y_train, y_test


