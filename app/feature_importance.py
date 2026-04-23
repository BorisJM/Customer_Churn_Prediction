import pandas as pd

def featureImportance(model, data):
     df = pd.DataFrame({
         "feature": data.columns,
         "coef": model.coef_[0]
     })

     df["abs_coef"] = df["coef"].abs()

     df.sort_values(by="abs_coef", ascending=False)
     # Print top 10
     print(df.head(10))
