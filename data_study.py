import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    df['MSE'] = df['MSE'] * 100
    df['Chi2pvalue'] = df['Chi2pvalue'].fillna(0)
    meandf = df.groupby(['n', 'N'], as_index=False).mean(numeric_only=True)
    print(meandf)