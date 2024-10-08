import collections

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from python_code import conf
from python_code.datasets.channels.sed_channel import SEDChannel
from python_code.evaluator import Evaluator
from python_code.utils.constants import Phase

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

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
        plt.ylabel(r'SNR [dB]')
        plt.xlabel(r'Block Index')
        plt.grid(True, which='both')
    plt.legend(loc='upper left')
    plt.show()
