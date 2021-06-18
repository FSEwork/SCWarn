import pandas as pd
import time, datetime

def add_ts(df):

    dates = df['datetime'].values
    ts = []
    for i in dates:
        timeArray = time.strptime(i, "%Y-%m-%d %H:%M:%S")
        ts.append(int(time.mktime(timeArray)))
    df['timestamp'] = ts
    print(df.head())
    return df

def add_datetime(df):
    ts = df['timestamp'].values
    dates = []
    for i in ts:
        timeArray = time.localtime(i)
        dates.append(time.strftime("%Y-%m-%d %H:%M:%S", timeArray))
    df['datetime'] = dates
    return df

def impute_missing(df):
    print(len(df))
    df = df.drop_duplicates(['timestamp'])
    ts = df['timestamp'].values
    for i in range(int(min(ts)),int(max(ts)),60):
        if i not in ts:
            #print(i)
            if i-min(ts) >= 1440:
                temp = df[df['timestamp'] == i-1440]
            else:
                temp = df[df['timestamp'] == i - 60]
            temp['timestamp'] = i
            temp['datetime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(i))
            df = df.append(temp)
    df = df.sort_values(by='timestamp')
    print(len(df))
    return df

def fill_nan(df: pd.DataFrame):
    df = df.copy()
    ts = df.index
    cols = df.columns

    begin_time = ts[0]
    end_time = ts[-1]
    strade = ts[1] - ts[0]

    for i in ts:
        for col in cols:
            if not pd.isna(df.loc[i, col]): continue

            if i - begin_time > pd.Timedelta('1day') and (not pd.isna(df.loc[i-pd.Timedelta('1day'), col])):
                df.loc[i, col] = df.loc[i-pd.Timedelta('1day'), col]
            else:
                df.loc[i, col] = df.loc[i-strade, col]
    return df



