import collections

import numpy as np
from matplotlib import pyplot as plt

from python_code import conf
from python_code.datasets.channels.sed_channel import SEDChannel
from python_code.evaluator import Evaluator
from python_code.utils.constants import Phase

if __name__ == "__main__":
    evaluator = Evaluator()
    sed_channel = SEDChannel()
    phase = Phase.TEST
    message_words, _ = evaluator.test_channel_dataset.__getitem__()

    plt.figure()
    snrs = collections.defaultdict(list)
    for t in range(conf.test_blocks_num):
        n_users = message_words[t].shape[1]
        cur_snrs = sed_channel.get_snrs(n_users, t, phase)
        for user in range(n_users):
            snrs[user].append([t, cur_snrs[user]])

    for user in snrs.keys():
        cur_samples = np.array(snrs[user])
        plt.plot(cur_samples[:, 0], cur_samples[:, 1], label=f'User {user + 1}')
        plt.ylabel(r'SNR [dB]', fontsize=20)
        plt.xlabel(r'Block Index', fontsize=20)
        plt.grid(True, which='both')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.show()
