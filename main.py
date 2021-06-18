import pandas as pd
import yaml
import argparse
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from approach.LSTM.lstm import *
from approach.LSTM.MLSTM import *
from approach.AutoEncoder.AE import *
from approach.AutoEncoder.VAE import *
from approach.AutoEncoder.MMAE import *
from approach.Metrics.ISST import ISST_predict
from approach.GRU.GRU import *
import torch
import os


# Global Configuration
with open("config.yml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# print(config)

# parse args
parser = argparse.ArgumentParser()
args = parser.parse_args()


def load_data(data_path):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # normalize
    if config['scaler'] == 'minmax':
        df = use_minmax_scale(df)
    if config['scaler'] == 'standard':
        df = use_standard_scale(df)
    return df


def use_minmax_scale(df_data: pd.DataFrame):
    df_data = df_data.copy()
    df_data.loc[:, :] = minmax_scale(df_data)
    return df_data


def use_standard_scale(df_data: pd.DataFrame):
    df_data = df_data.copy()
    scaler = StandardScaler()
    df_data.loc[:, :] = scaler.fit_transform(df_data)
    return  df_data


def train_model(algorithm, train_data: np.ndarray, params):
    if algorithm == "LSTM":
        model = get_model_LSTM(train_data, params['seq_len'], params['batch_size'],
                               params['epoch'], params['learning_rate'])
    elif algorithm == "GRU":
        model = get_model_GRU(train_data, params['seq_len'], params['batch_size'],
                               params['epoch'], params['learning_rate'])
    elif algorithm == "MLSTM":
        model = get_model_MLSTM(train_data, params['modal'],
                                params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
    elif algorithm == "AE":
        model = get_model_AE(train_data, params['seq_len'], params['batch_size'],
                             params['epoch'], params['learning_rate'])
    elif algorithm == "MMAE":
        model = get_model_MMAE(train_data, params['modal'],
                                params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
    elif algorithm == "VAE":
        model = get_model_VAE(train_data, params['seq_len'], params['batch_size'],
                              params['epoch'], params['learning_rate'])
    elif algorithm == "GRU":
        model = get_model_GRU(train_data, params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])

    torch.save(model, f"model/{algorithm}_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")


def run_lstm(train_data: np.ndarray, test_data: np.ndarray, params):
    if 'model_path' in params:
        if os.path.exists(params['model_path']):
            model = torch.load(params['model_path'])
        else:
            raise FileNotFoundError(f"'{params['model_path']}' can not be found. Please check the model_path parameter in config.yaml")
    else:
        model = get_model_LSTM(train_data, params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
        torch.save(model, f"model/LSTM_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

    scores, dim_scores = get_prediction_LSTM(model, test_data, params['seq_len'])

    return scores

def run_gru(train_data: np.ndarray, test_data: np.ndarray, params):
    if 'model_path' in params:
        model = torch.load(params['model_path'])
    else:
        model = get_model_GRU(train_data, params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
        torch.save(model, f"model/GRU_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

    scores, dim_scores = get_prediction_GRU(model, test_data, params['seq_len'])

    return scores


def run_mlstm(train_data: np.ndarray, test_data: np.ndarray, params):
    if 'model_path' in params:
        model = torch.load(params['model_path'])
    else:
        model = get_model_MLSTM(train_data, params['modal'],
                                params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
        torch.save(model, f"model/MLSTM_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

    scores, dim_scores = get_prediction_MLSTM(model, test_data, params['seq_len'], params['modal'])

    return scores


def run_ae(train_data: np.ndarray, test_data: np.ndarray, params):
    if 'model_path' in params:
        model = torch.load(params['model_path'])
    else:
        model = get_model_AE(train_data, params['seq_len'], params['batch_size'],
                             params['epoch'], params['learning_rate'])
        torch.save(model, f"model/AE_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

    scores, dim_scores = get_prediction_AE(model, test_data, params['seq_len'])
    return scores


def run_mmae(train_data: np.ndarray, test_data: np.ndarray, params):
    if 'model_path' in params:
        model = torch.load(params['model_path'])
    else:
        model = get_model_MMAE(train_data, params['modal'],
                               params['seq_len'], params['batch_size'], params['epoch'], params['learning_rate'])
        torch.save(model, f"model/MMAE_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

    scores, dim_scores = get_prediction_MMAE(model, test_data, params['seq_len'], params['modal'])
    return scores


def run_vae(train_data: np.ndarray, test_data: np.ndarray, params):
    if 'model_path' in params:
        model = torch.load(params['model_path'])
    else:
        model = get_model_VAE(train_data, params['seq_len'], params['batch_size'],
                              params['epoch'], params['learning_rate'])
        torch.save(model, f"model/VAE_{params['epoch']}_{params['batch_size']}_{params['learning_rate']}.pt")

    scores, dim_scores = get_prediction_VAE(model, test_data, params['seq_len'])
    return scores


def run_isst(train_data: np.ndarray, test_data: np.ndarray, params):
    try:
        scores = ISST_predict(test_data[:, params['dim_pos']])
    except:
        print("Error")
        return [0] * test_data.shape[0]
    return scores


def run_algorithms(algorithms, train_data: np.ndarray, test_data: np.ndarray):
    results = {}
    for i in algorithms:
        run = None
        if i == "LSTM":
            run = run_lstm
        elif i == "MLSTM":
            run = run_mlstm
        elif i == "AE":
            run = run_ae
        elif i == "MMAE":
            run = run_mmae
        elif i == "VAE":
            run = run_vae
        elif i == "ISST":
            run = run_isst
        elif i == "GRU":
            run = run_gru

        if run is not None:
            scores = run(train_data, test_data, algorithms[i])
            results[i] = [float(i) for i in scores]

            if 'seq_len' in algorithms[i]:
                seq_len = algorithms[i]['seq_len']
                results[i] = [np.nan] * seq_len + results[i]

        # incorrect configuration
        else:
            print(f"{i} isn't included in SCWarn. Please check the config.yml.")

    return results


def test_case(df_train: pd.DataFrame, df_test: pd.DataFrame):
    df_train, df_test = df_train.copy(), df_test.copy()

    # run algorithms
    train_data, test_data = df_train.to_numpy(), df_test.to_numpy()
    results = run_algorithms(config['algorithms'], train_data, test_data)

    # save results
    df = pd.DataFrame(results)
    df['timestamp'] = df_test.index
    df = df.set_index('timestamp')

    return df


if __name__ == '__main__':
    # load training data
    os.makedirs("model", exist_ok=True)
    df_train = load_data(config['train_path'])

    # train and test
    if 'test_path' in config:
        multi_cases = True if os.path.isdir(config['test_path']) else False

        if not multi_cases:
            df_test = load_data(config['test_path'])
            df = test_case(df_train, df_test)

        else:
            test_info_ls = [(load_data(os.path.join(config['test_path'], i)), i)
                            for i in os.listdir(config['test_path']) if i.endswith(".csv")]
            result_ls = []
            for df_test, case_name in test_info_ls:
                tmp_df = test_case(df_train, df_test)
                tmp_df['case'] = case_name[:-4]
                result_ls.append(tmp_df)
            df = pd.concat(result_ls)

        os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
        df.to_csv(config['output_path'])

    # barely train
    else:
        for algorithm in config['algorithms']:
            train_model(algorithm, df_train.to_numpy(), config['algorithms'][algorithm])
