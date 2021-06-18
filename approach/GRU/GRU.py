from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from util.dataset import use_mini_batch, apply_sliding_window


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size=1):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hiddent2out = nn.Linear(hidden_dim, input_dim)

    def forward(self, seq):
        gru_out, _ = self.gru(seq.view(self.batch_size, -1, self.input_dim))
        predict = self.hiddent2out(gru_out)
        return predict[:, -1, :]


def train(dataloader, input_dim, batch_size, n_epoch, lr=0.01):
    model = GRU(input_dim, 100, batch_size)  #type: GRU
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        t0 = time()
        print("epoch: %d / %d" % (epoch+1, n_epoch))

        loss_sum = 0
        for step, (batch_X, batch_Y) in enumerate(dataloader):
            model.zero_grad()
            predicted = model(batch_X)
            loss = loss_function(predicted, batch_Y)

            loss_sum += loss.item()
            if step % 100 == 0:
                print(loss_sum / 100)
                loss_sum = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("time: %.2f s" % float(time() - t0))
    return model

def predict(model, test_data, seq_len):
    model.batch_size = 1
    predict_ls = []
    anomaly_scores = []
    loss_function = nn.MSELoss()

    anomaly_scores_per_dim = []
    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            seq = torch.tensor(test_data[i-seq_len : i]).float()
            predicted = model(seq)[0]

            ground_truth = torch.tensor(test_data[i]).float()
            anomaly_score = loss_function(predicted, ground_truth)

            predict_ls.append(predicted.tolist())
            anomaly_scores.append(anomaly_score)

            anomaly_scores_per_dim.append(np.abs(predicted.numpy() - ground_truth.numpy()))

    anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)
    # anomaly_scores = minmax_scale(anomaly_scores)

    return predict_ls, anomaly_scores, anomaly_scores_per_dim


def get_model_GRU(train_data: np.ndarray, seq_len: int, batch_size: int, n_epoch, lr=0.01):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len, flatten=False)
    train_data_loader = use_mini_batch(seq_dataset, seq_ground_truth, batch_size)

    input_dim = train_data_loader.dataset.feature_len

    model = train(train_data_loader, input_dim, batch_size, n_epoch, lr)

    return model


def get_prediction_GRU(model, test_data, seq_len):
    predict_ls, scores, dim_scores = predict(model, test_data, seq_len)

    return scores, dim_scores
