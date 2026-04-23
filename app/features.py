import pandas as pd

def createFeatures(df):
    df = df.copy()

    # Delete customerID from the data
    df.drop("customerID", axis=1, inplace=True)
    # Total charges conversion to number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df.dropna(inplace=True)