from sklearn.preprocessing import minmax_scale
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from time import time
from util.dataset import use_mini_batch, apply_sliding_window
from util.corrloss import CorrLoss

class MLSTM(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, batch_size=64):
        super(MLSTM, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.lstm1 = nn.LSTM(input_dim1, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim2, hidden_dim, batch_first=True)

        self.hiddent1out = nn.Linear(hidden_dim*2, input_dim1)
        self.hiddent2out = nn.Linear(hidden_dim*2, input_dim2)

    def forward(self, seq1, seq2):
        lstm_out1, _ = self.lstm1(seq1.view(self.batch_size, -1, self.input_dim1))
        lstm_out2, _ = self.lstm2(seq2.view(self.batch_size, -1, self.input_dim2))
        shared = torch.cat((lstm_out1, lstm_out1), 2)

        # print(lstm_out1.shape, lstm_out2.shape, shared.shape)  #torch.Size([128, 20, 8])

        predict1 = self.hiddent1out(shared)
        predict2 = self.hiddent2out(shared)
        predict = torch.cat((predict1, predict2), 2)

        #print(predict1.shape, predict2.shape, predict.shape) #torch.Size([128, 10, 4]) torch.Size([128, 10, 7]) torch.Size([128, 10, 11])

        return predict[:, -1, :], (lstm_out1[:, -1, :], lstm_out2[:, -1, :])


def train(dataloader, modal, batch_size, n_epoch, lr=0.01):
    input_dim1, input_dim2 = modal[0], modal[1]
    model = MLSTM(input_dim1, input_dim2, 8, batch_size)
    loss_function = nn.MSELoss()
    loss_corr = CorrLoss()

    optimizer1 = optim.SGD(model.parameters(), lr=lr)


    for epoch in range(n_epoch):
        t0 = time()
        print("epoch: %d / %d" % (epoch+1, n_epoch))

        loss_sum = 0
        for step, (batch_X, batch_Y) in enumerate(dataloader):
            model.zero_grad()
            #print(batch_X.shape,batch_Y.shape)   #torch.Size([128, 10, 11]) torch.Size([128, 11])
            predicted, (H1, H2) = model(batch_X[:,:, :input_dim1], batch_X[:, :, input_dim1:])
            # loss = loss_function(predicted, batch_Y) + loss_corr(H1, H2)
            loss = loss_function(predicted, batch_Y)
            loss_sum += loss.item()
            if step % 100 == 0:
                print(loss_sum / 100)
                loss_sum = 0

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        print("time: %.2f s" % float(time() - t0))
    return model


def predict(model, test_data, seq_len, modal):
    model.batch_size = 1
    predict_ls = []
    anomaly_scores = []
    loss_function = nn.MSELoss()

    anomaly_scores_per_dim = []
    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            seq = torch.tensor(test_data[i-seq_len : i]).float()

            predicted, _ = model(seq[:,:modal[0]], seq[:,modal[0]:])

            ground_truth = torch.tensor(test_data[i]).float()
            anomaly_score = loss_function(predicted.view(-1), ground_truth.view(-1))

            predict_ls.append(predicted.tolist())
            anomaly_scores.append(anomaly_score)

            anomaly_scores_per_dim.append(np.abs(predicted.numpy() - ground_truth.numpy()))

    anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)
    # anomaly_scores = minmax_scale(anomaly_scores)

    return predict_ls, anomaly_scores, anomaly_scores_per_dim

def get_model_MLSTM(train_data, modal, seq_len=10, batch_size=64, n_epoch=10, lr=0.01):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len,flatten=False)
    train_data_loader = use_mini_batch(seq_dataset, seq_ground_truth, batch_size)

    model = train(train_data_loader, modal, batch_size, n_epoch, lr)

    return model

def get_prediction_MLSTM(model, test_data, seq_len, modal):
    predict_ls, scores, dim_scores = predict(model, test_data, seq_len, modal)
    return scores, dim_scores


def run_mlstm(train_data, test_data, modal, seq_len=10, batch_size=64, n_epoch=10):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len,flatten=False)
    train_data_loader = use_mini_batch(seq_dataset, seq_ground_truth, batch_size)

    # input_dim = train_data_loader.dataset.feature_len
    # print(input_dim)

    model = train(train_data_loader, modal, batch_size, n_epoch)
    predict_ls, scores, dim_scores = predict(model, test_data, seq_len, modal)

    return scores, dim_scores