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
        {'detector_type': 'rec_deepsic', 'training_type': 'Joint'},
        {'detector_type': 'rec_deepsic', 'training_type': 'Online'},
        {'detector_type': 'hyper_deepsic', 'training_type': 'Joint'},
    ]
    pilot_sizes = [400, 800, 1200, 1600, 2000]
    seeds = range(1, 4)

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
        for seed in seeds:
            conf.set_value('seed', seed)
            cur_ser_values = []
            evaluator = Evaluator()
            method_name = evaluator.detector.__str__()
            for pilot_size in pilot_sizes:
                conf.set_value('test_pilots_length', pilot_size)
                metrics_output: MetricOutput = evaluator.evaluate()
                cur_ser = np.mean(np.array(metrics_output.ser_list))
                cur_ser_values.append(cur_ser)
            ser_values.append(cur_ser_values)
        ser_values = np.sum(np.array(ser_values), axis=0) / len(seeds)
        plt.plot(pilot_sizes, ser_values, label=method_name, color=COLORS_DICT[method_name],
                 marker=MARKERS_DICT[method_name], markersize=11,
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2)
    plt.xlabel('Pilots Number')
    plt.ylabel('SER')
    plt.grid(which='both', ls='--')
    leg = plt.legend(loc='lower left', prop={'size': 20}, handlelength=4)
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_vs_pilots_number_{conf.n_ant}.png'),
                bbox_inches='tight')
    plt.show()