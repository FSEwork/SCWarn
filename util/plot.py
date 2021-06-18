import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

def plot_data(data, dataname, scores, methods, columns, xdate=None):
    try:
        xs = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in xdate]
    except:
        xs = xdate
    n_sample, n_dimension = data.shape[0], data.shape[1]
    fig, ax = plt.subplots(n_dimension + len(scores), 1, figsize=(12, (n_dimension + len(scores)) * 2))
    for i in range(n_dimension):
        if xdate is not None:
            ax[i].plot(xs, data[:, i], color='black', linewidth=0.7)
        else:
            ax[i].plot(data[:, i], color='black', linewidth=0.7)
        ax[i].set_title(columns[i])
        ax[i].set_yticks([])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for i in range(len(scores)):
        if xdate is not None:
            ax[n_dimension + i].plot(xs, list(np.zeros(len(data) - len(scores[i]))) + list(scores[i]),
                                     color='darkorange')
        else:
            ax[n_dimension + i].plot(list(np.zeros(len(data) - len(scores[i]))) + list(scores[i]), color='darkorange')
        ax[n_dimension + i].set_title(methods[i])
        ax[n_dimension + i].set_yticks([])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.tight_layout(pad=0)
    plt.subplots_adjust(hspace=1)
    plt.savefig('result/' + dataname + '.png')


def plot_result(dataname, scores, methods, xdate=None):
    try:
        xs = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in xdate]
    except:
        xs = xdate

    fig, ax = plt.subplots(len(scores), 1, figsize=(12, (len(scores)) * 2))

    for i in range(len(scores)):
        if xdate is not None:
            ax[i].plot(xs, list(np.zeros(len(xs) - len(scores[i]))) + list(scores[i]), color='darkorange')
        else:
            ax[i].plot(list(np.zeros(len(xs) - len(scores[i]))) + list(scores[i]), color='darkorange')
        ax[i].set_title(methods[i])
        ax[i].set_yticks([])
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.tight_layout(pad=0)
    plt.subplots_adjust(hspace=1)
    plt.savefig('result/' + dataname + '-result.png')


# def plot_result(test_data, total_score, methods, dataname):
#     '''
#     :param test_data: n_sample * n_dimension
#     :param total_score: list, length = #methods, n_sample * 1
#     :return: a figure
#     '''
#     n_sample, n_dimension = test_data.shape[0], test_data.shape[1]
#
#     fig, ax = plt.subplots(n_dimension + len(total_score), 1, figsize=(8, (n_dimension + len(total_score)) * 2))
#     for i in range(n_dimension):
#         ax[i].plot(test_data[:, i])
#         # ax2 = ax[i].twinx()
#         # ax2.plot(dim_score[:, i], color='pink', linewidth=0.9)
#         # ax2.set_ylim(min(dim_score[:,i]),max(dim_score[:,i])*2)
#
#         # ax[i].set_yticks([])
#
#     index = 0
#     for i in total_score:
#         ax[n_dimension + index].plot(i, color='pink')
#         ax[n_dimension + index].set_xlabel(methods[index])
#         index += 1
#
#     plt.tight_layout(pad=0)
#     plt.savefig('result/' + dataname + methods[0] + '_result.png')
#     # plt.show()
