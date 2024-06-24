import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from dir_definitions import FIGURES_DIR
from python_code import conf
from python_code.evaluator import Evaluator, MetricOutput
from python_code.plotting import *

if __name__ == "__main__":
    params_list = [
        {'detector_type': 'rec_deepsic', 'training_type': 'Joint', 'train_block_length': 1000},
        {'detector_type': 'rec_deepsic', 'training_type': 'Online'},
        {'detector_type': 'hyper_deepsic', 'training_type': 'Joint', 'train_block_length': 5000},
    ]
    COST_snrs = [8, 10, 12, 14]
    seeds = [1, 2, 3, 4, 5]

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

        ser_values = []
        for snr in COST_snrs:
            conf.set_value('COST_SNR', snr)
            cur_ser = 0
            for seed in seeds:
                conf.set_value('seed', seed)
                evaluator = Evaluator()
                metrics_output: MetricOutput = evaluator.evaluate()
                method_name = evaluator.detector.__str__()
                cur_ser += np.mean(np.array(metrics_output.ser_list))
            ser_values.append(cur_ser / len(seeds))
        plt.plot(COST_snrs, ser_values, label=method_name, color=COLORS_DICT[method_name],
                 markevery=20, marker=MARKERS_DICT[method_name], markersize=11,
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2)

    plt.xlabel('SNR [dB]')
    plt.ylabel('SER')
    plt.grid(which='both', ls='--')
    leg = plt.legend(loc='upper left', prop={'size': 20}, handlelength=4)
    plt.yscale('log')
    plt.ylim(bottom=5 * 10 ** -4, top=5 * 10 ** -2)
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_vs_cost_snr_{conf.n_ant}.png'),
                bbox_inches='tight')
    plt.show()
