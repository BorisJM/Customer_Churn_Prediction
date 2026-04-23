import pandas as pd

def encodingDatabase(df):
    df = pd.get_dummies(df, drop_first=True)

    return df