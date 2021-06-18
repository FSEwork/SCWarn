import pandas as pd
import numpy as np

def load_seq_files(file_name):
    with open(file_name, 'r') as f:
        lines = [i.strip().split() for i in f if i]

    seq = np.array([[int(i[0]), int(i[1])] for i in lines])

    return seq

def to_dataframe(seq, unit='us'):
    df = pd.DataFrame(seq, columns=['Time', 'ID'])
    df['Time'] = pd.to_datetime(df['Time'], unit=unit)
    df.set_index('Time', inplace=True)
    
    return df

def group_by_id(seq_df, resample='H'):
    """
    return: dictionary of tempaltes serires. Keys are the templates' ids and values are dataframes of each serires' count.
    """
    group_dict = {}

    # set of templates id
    id_set = set(seq_df['ID'].values)

    for i in id_set:
        one_templates_series = seq_df[seq_df['ID'] == i]
        count_series = one_templates_series.resample(resample).count()
        count_series.rename(columns={'ID': str(i)}, inplace=True)
        
        # drop templates whose total count is less than 100
        if count_series.count().values > 100:
            group_dict[i] = count_series
    
    return group_dict

def gen_group_dict(log_file_path, unit='us', resample='H'):
    seq = load_seq_files(log_file_path)
    df = to_dataframe(seq, unit)
    return group_by_id(df, resample)


def gen_group_df(log_file_path, unit='us', resample='H'):
    group_dict = gen_group_dict(log_file_path, unit, resample)
    df = pd.concat(list(group_dict.values()), axis=1)
    df.fillna(0, inplace=True)

    return df
    
# if __name__ == '__main__':
#     seq = load_seq_files("./train_log_seqence.txt")
#     df = to_dataframe(seq)
#     group_dict = group_by_id(df)
#
#     print(len(group_dict))
#     print(group_dict.keys())
#     print(group_dict[1])
