def setTarget(df):
    # Change Yes to 1 and No to 0 in Churn column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df