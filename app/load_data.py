import pandas as pd

def loadData(path):
    # Read data
    df = pd.read_csv(path)

    return df

