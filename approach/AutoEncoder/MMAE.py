import torch
from time import time
from torch import nn
import numpy as np
from torch.autograd import Variable
from sklearn.preprocessing import minmax_scale
from util.dataset import use_mini_batch, apply_sliding_window
from util.corrloss import CorrLoss


class MultiModalAutoencoder(nn.Module):

    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim1, output_dim2):
        super(MultiModalAutoencoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim1, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim2, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim1),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim2),
        )

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        shared = torch.cat((x1, x2), 1)
        d_x1 = self.decoder1(shared)
        d_x2 = self.decoder2(shared)
        x = torch.cat((d_x1, d_x2), 1)
        return x, (x1, x2)


def train(dataloader0, dataloader1, modal, n_epoch, lr=0.001):
    output_dim1, input_dim1 = modal[0], modal[0]
    output_dim2, input_dim2 = modal[1], modal[1]

    model = MultiModalAutoencoder(input_dim1, input_dim2, 8, output_dim1, output_dim2)

    loss_function, corr_loss = nn.MSELoss(), CorrLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(n_epoch):
        t0 = time()
        print("epoch: %d / %d" % (epoch + 1, n_epoch))

        loss_sum = 0
        # for step, (batch_X, batch_Y) in enumerate(dataloader):
        for step, batch_data in enumerate(zip(dataloader0, dataloader1)):
            batch_X0, batch_Y0 = batch_data[0]
            batch_X1, batch_Y1 = batch_data[1]
            batch_Y = torch.cat((batch_Y0, batch_Y1), 1)

            model.zero_grad()
            # print(batch_X)
            # print(batch_X.size())
            predict, (x1, x2) = model(batch_X0, batch_X1)
            # loss = loss_function(predict, batch_Y) - 0.1*corr_loss(x1, x2)
            loss = loss_function(predict, batch_Y)
            loss_sum += loss.item()
            if (step + 1) % 100 == 0:
                print(loss_sum / 100)
                loss_sum = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("time: %.2f s" % float(time() - t0))
    return model

def predict(model, test_data, seq_len, modal):
    predict_ls = []
    anomaly_scores = []
    loss_function = nn.MSELoss()

    anomaly_scores_per_dim = []
    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            test_data0, test_data1 = test_data[:, :modal[0]], test_data[:, modal[0]:]
            seq0 = torch.tensor(test_data0[i - seq_len: i].reshape(1, -1)).float()
            seq1 = torch.tensor(test_data1[i - seq_len: i].reshape(1, -1)).float()

            # print(seq0.size())
            # print(seq1.size())

            predict, _ = model(seq0, seq1)

            anomaly_score = loss_function(predict, torch.cat((seq0, seq1), 1))

            predict_ls.append(predict.tolist())
            anomaly_scores.append(anomaly_score)

            # TODO
            # anomaly_scores_per_dim.append(np.abs(predict.numpy() - seq.numpy()))

    anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)
    # anomaly_scores = minmax_scale(anomaly_scores)
    return predict_ls, anomaly_scores, anomaly_scores_per_dim


def get_model_MMAE(train_data, modal, seq_len=10, batch_size=64, n_epoch=10, lr=0.001):
    seq_dataset_0, seq_ground_truth_0 = apply_sliding_window(train_data[:, :modal[0]], seq_len=seq_len, flatten=True)
    seq_dataset_1, seq_ground_truth_1 = apply_sliding_window(train_data[:, modal[0]:], seq_len=seq_len, flatten=True)
    train_data_loader_0 = use_mini_batch(seq_dataset_0, seq_dataset_0, batch_size)
    train_data_loader_1 = use_mini_batch(seq_dataset_1, seq_dataset_1, batch_size)

    modal_train = [i * seq_len for i in modal]
    print(modal)

    model = train(train_data_loader_0, train_data_loader_1, modal_train, n_epoch, lr)

    return model


def get_prediction_MMAE(model, test_data, seq_len, modal):
    predict_result, anomaly_score, dim_score = predict(model, test_data, seq_len, modal)
    return anomaly_score, dim_score


def run_mmae(train_data, test_data, modal, seq_len=10, batch_size=64, n_epoch=10):
    seq_dataset_0, seq_ground_truth_0 = apply_sliding_window(train_data[:, :modal[0]], seq_len=seq_len, flatten=True)
    seq_dataset_1, seq_ground_truth_1 = apply_sliding_window(train_data[:, modal[0]:], seq_len=seq_len, flatten=True)
    train_data_loader_0 = use_mini_batch(seq_dataset_0, seq_dataset_0, batch_size)
    train_data_loader_1 = use_mini_batch(seq_dataset_1, seq_dataset_1, batch_size)

    #input_dim = train_data_loader.dataset.feature_len
    modal_train = [i * seq_len for i in modal]
    print(modal)

    model = train(train_data_loader_0, train_data_loader_1, modal_train, n_epoch)

    predict_result, anomaly_score, dim_score = predict(model, test_data, seq_len, modal)

    return anomaly_score, dim_score
