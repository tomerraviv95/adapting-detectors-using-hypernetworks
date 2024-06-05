import os
from datetime import datetime

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from dir_definitions import FIGURES_DIR
from python_code import conf
from python_code.evaluator import Evaluator, MetricOutput

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 15
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

COLORS_DICT = {'Joint Hypernetwork-based DeepSIC': 'red',
               'Joint Recurrent DeepSIC': 'orange',
               'Online Recurrent DeepSIC': 'blue',
               'Online Sequential DeepSIC': 'green',
               'Joint RNN Hypernetwork-based DeepSIC': 'black'}

MARKERS_DICT = {'Joint Hypernetwork-based DeepSIC': 'd',
                'Joint Recurrent DeepSIC': 'd',
                'Online Recurrent DeepSIC': 'x',
                'Online Sequential DeepSIC': 'x',
                'Joint RNN Hypernetwork-based DeepSIC': 'o'}

LINESTYLES_DICT = {'Joint Hypernetwork-based DeepSIC': 'solid',
                   'Joint Recurrent DeepSIC': 'dotted',
                   'Online Recurrent DeepSIC': 'solid',
                   'Online Sequential DeepSIC': 'dotted',
                   'Joint RNN Hypernetwork-based DeepSIC': 'solid'}

if __name__ == "__main__":
    params_list = [
        {'detector_type': 'rec_deepsic', 'training_type': 'Joint', 'train_block_length': 1000},
        {'detector_type': 'rec_deepsic', 'training_type': 'Online'},
        {'detector_type': 'hyper_deepsic', 'training_type': 'Joint', 'train_block_length': 5000},
    ]
    seeds = [1, 2, 3]

    # path for the saved figure
    current_day_time = datetime.now()
    folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
    if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
        os.makedirs(os.path.join(FIGURES_DIR, folder_name))

    # extract names from simulated plots
    plt.figure()
    for params in params_list:
        for key, value in params.items():
            conf.set_value(key, value)
        values = 0
        for seed in seeds:
            conf.set_value('seed', seed)
            evaluator = Evaluator()
            metrics_output: MetricOutput = evaluator.evaluate()
            method_name = evaluator.detector.__str__()
            values += np.cumsum(np.array(metrics_output.ser_list)) / len(metrics_output.ser_list)
        values /= len(seeds)
        plt.plot(range(len(metrics_output.ser_list)), values, label=method_name, color=COLORS_DICT[method_name],
                 markevery=20, marker=MARKERS_DICT[method_name], markersize=11,
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2)

    plt.xlabel('Block Index')
    plt.ylabel('SER')
    plt.grid(which='both', ls='--')
    leg = plt.legend(loc='upper left', prop={'size': 20}, handlelength=4)
    plt.yscale('log')
    plt.ylim(bottom=5 * 10 ** -4, top=3 * 10 ** -2)
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_{conf.n_ant}.png'),
                bbox_inches='tight')
    plt.show()
