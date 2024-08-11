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
        {'detector_type': 'joint_deepsic'},
        {'detector_type': 'online_deepsic'},
        {'detector_type': 'hyper_deepsic'},
    ]
    seeds = range(1, 2)

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
            method_name = evaluator.detector.__str__()
            metrics_output: MetricOutput = evaluator.evaluate()
            values += np.cumsum(np.array(metrics_output.ser_list)) / len(metrics_output.ser_list)
        values /= len(seeds)
        plt.plot(range(len(metrics_output.ser_list)), values, label=method_name, color=COLORS_DICT[method_name],
                 markevery=20, marker=MARKERS_DICT[method_name], markersize=11,
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2)

    plt.xlabel('Block Index')
    plt.ylabel('SER')
    plt.grid(which='both', ls='--')
    leg = plt.legend(loc='lower right', prop={'size': 20}, handlelength=4)
    plt.yscale('log')
    plt.ylim(bottom=1e-4)
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_{conf.n_ant}.png'),
                bbox_inches='tight')
    plt.show()
