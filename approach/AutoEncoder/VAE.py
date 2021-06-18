import torch
from time import time
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.preprocessing import minmax_scale
from util.dataset import use_mini_batch, apply_sliding_window


class VAE(nn.Module):
    def __init__(self,input_dim, hidden_dim, out_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, hidden_dim)
        self.fc22 = nn.Linear(128, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.fc4 = nn.Linear(128, out_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(dataloader, input_dim, n_epoch, lr=0.001):
    output_dim = input_dim
    model = VAE(input_dim, 4, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(n_epoch):
        t0 = time()
        print("epoch: %d / %d" % (epoch + 1, n_epoch))
        loss_sum = 0

        for step, (batch_X, batch_Y) in enumerate(dataloader):
            model.zero_grad()

            recon, mu, logvar = model(batch_X)

            loss = loss_function(recon, batch_X, mu, logvar)
            if (step + 1) % 100 == 0:
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
    anomaly_scores_per_dim = []
    with torch.no_grad():
        for i in range(seq_len, len(test_data)):
            seq = torch.tensor(test_data[i - seq_len: i].reshape(1,-1)).float()
            recon, mu, logvar = model(seq)

            anomaly_score = loss_function(recon, seq, mu, logvar).item()
            predict_ls.append(recon.tolist())
            anomaly_scores.append(anomaly_score)

            anomaly_scores_per_dim.append(np.abs(recon.numpy() - seq.numpy()))

    anomaly_scores_per_dim = np.array(anomaly_scores_per_dim)
    # anomaly_scores = minmax_scale(anomaly_scores)
    return predict_ls, anomaly_scores, anomaly_scores_per_dim


def get_model_VAE(train_data, seq_len=10, batch_size=64, n_epoch=10, lr=0.01):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len, flatten=True)
    train_data_loader = use_mini_batch(seq_dataset, seq_dataset, batch_size)
    input_dim = train_data_loader.dataset.feature_len
    model = train(train_data_loader, input_dim, n_epoch, lr)

    return model


def get_prediction_VAE(model, test_data, seq_len):
    predict_result, anomaly_score, dim_score = predict(model, test_data, seq_len)
    return anomaly_score, dim_score


def run_vae(train_data, test_data, seq_len=10, batch_size=64, n_epoch=10):
    seq_dataset, seq_ground_truth = apply_sliding_window(train_data, seq_len=seq_len, flatten=True)
    train_data_loader = use_mini_batch(seq_dataset, seq_dataset, batch_size)
    input_dim = train_data_loader.dataset.feature_len
    model = train(train_data_loader, input_dim, n_epoch)

    predict_result, anomaly_score, dim_score = predict(model, test_data, seq_len)
    return anomaly_score, dim_score