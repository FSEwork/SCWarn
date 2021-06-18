import os
import pandas as pd
import numpy as np

def ISST_predict(test_data: np.ndarray):
    timestamps = [i for i in range(test_data.shape[0])]
    df_data = pd.DataFrame({'timestamp': timestamps, 'value': test_data * 100})
    df_data.to_csv("isst_tmp.csv", index=False)
    os.system("approach/Metrics/ISST.out isst_tmp.csv")
    df_scores = pd.read_csv("cd_isst_tmp.csv", header=None)
    scores = df_scores[1].tolist()
    os.system("rm cd_isst_tmp.csv isst_tmp.csv")
    return [np.nan] * 29 + scores + [np.nan] * 30