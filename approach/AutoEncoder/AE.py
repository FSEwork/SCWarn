import torch
from time import time
from torch import nn
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import minmax_scale
from util.dataset import use_mini_batch, apply_sliding_window


class Autoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(dataloader, input_dim, n_epoch, lr=0.001):

    output_dim = input_dim
    model = Autoencoder(input_dim, 16, output_dim)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(n_epoch):
        t0 = time()
        print("epoch: %d / %d" % (epoch + 1, n_epoch))

        loss_sum = 0
        for step, (batch_X, batch_Y) in enumerate(dataloader):
            model.zero_grad()
            predicted = model(batch_X)
            loss = loss_function(predicted, batch_Y)

            loss_sum += loss.item()
            if (step+1) % 100 == 0:
                print(loss_sum / 100)
                loss_sum = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("time: %.2f s" % float(time() - t0))
    return model


def predict(model, test_data, seq_len):
    predict_ls = []
    anomaly_scores = []
    loss_function = nn.MSELoss()

    anomaly_scores_per_dim = []
    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            seq = torch.tensor(test_data[i - seq_len: i].reshape(1,-1)).float()
            predicted = model(seq)

            # ground_truth = torch.tensor(test_data[i]).float()
            anomaly_score = loss_function(predicted, seq)

            predict_ls.append(predicted.tolist())
            anomaly_scores.append(anomaly_score)

            anomaly_scores_per_dim.append(np.abs(predicted.numpy() - seq.numpy()))

    anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)
    # anomaly_scores = minmax_scale(anomaly_scores)
    return predict_ls, anomaly_scores, anomaly_scores_per_dim


def get_model_AE(train_data, seq_len=10, batch_size=64, n_epoch=10, lr=0.001):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len, flatten=True)
    train_data_loader = use_mini_batch(seq_dataset, seq_dataset, batch_size)
    input_dim = train_data_loader.dataset.feature_len

    model = train(train_data_loader, input_dim, n_epoch, lr)

    return model


def get_prediction_AE(model, test_data, seq_len):
    predict_result, anomaly_score, dim_score = predict(model, test_data, seq_len)

    return anomaly_score, dim_score


def run_ae(train_data, test_data, seq_len=10, batch_size=64, n_epoch=10):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len, flatten=True)
    train_data_loader = use_mini_batch(seq_dataset, seq_dataset, batch_size)
    input_dim = train_data_loader.dataset.feature_len

    model = train(train_data_loader, input_dim, n_epoch)

    predict_result, anomaly_score, dim_score = predict(model, test_data, seq_len)

    return anomaly_score, dim_score
