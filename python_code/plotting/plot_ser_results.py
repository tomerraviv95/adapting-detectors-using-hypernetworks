import os
from datetime import datetime

import matplotlib as mpl
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
               'Joint Sequential DeepSIC': 'black'}

MARKERS_DICT = {'Joint Hypernetwork-based DeepSIC': 'd',
                'Joint Recurrent DeepSIC': 'd',
                'Online Recurrent DeepSIC': 'x',
                'Online Sequential DeepSIC': 'x',
                'Joint Sequential DeepSIC': 'o'}

LINESTYLES_DICT = {'Joint Hypernetwork-based DeepSIC': 'solid',
                   'Joint Recurrent DeepSIC': 'dotted',
                   'Online Recurrent DeepSIC': 'solid',
                   'Online Sequential DeepSIC': 'dotted',
                   'Joint Sequential DeepSIC': 'solid'}

if __name__ == "__main__":
    params_list = [
        {'detector_type': 'seq_deepsic', 'training_type': 'Joint'},
        {'detector_type': 'seq_deepsic', 'training_type': 'Online'},
        {'detector_type': 'rec_deepsic', 'training_type': 'Joint'},
        {'detector_type': 'rec_deepsic', 'training_type': 'Online'},
        {'detector_type': 'hyper_deepsic', 'training_type': 'Joint'},
    ]

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
        evaluator = Evaluator()
        metrics_output: MetricOutput = evaluator.evaluate()
        method_name = evaluator.detector.__str__()
        plt.plot(range(len(metrics_output.ser_list)), metrics_output.ser_list, label=method_name,
                 color=COLORS_DICT[method_name],
                 marker=MARKERS_DICT[method_name], markersize=11,
                 linestyle=LINESTYLES_DICT[method_name], linewidth=2.2)

    plt.xlabel('Block Index')
    plt.ylabel('SER')
    plt.grid(which='both', ls='--')
    leg = plt.legend(loc='lower right', prop={'size': 15}, handlelength=4)
    plt.yscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, folder_name, f'ser_{conf.n_user}_{conf.n_ant}.png'),
                bbox_inches='tight')
    plt.show()
