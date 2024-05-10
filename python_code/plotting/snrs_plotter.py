from matplotlib import pyplot as plt

from python_code.evaluator import Evaluator
from python_code.utils.constants import Phase

if __name__ == "__main__":
    evaluator = Evaluator()
    PLOT_TRAIN = True
    if PLOT_TRAIN:
        _, _, snrs_list = evaluator.train_channel_dataset.__getitem__(phase=Phase.TRAIN)
    else:
        _, _, snrs_list = evaluator.test_channel_dataset.__getitem__(phase=Phase.TEST)

    plt.figure()

    for i in range(len(snrs_list[0])):
        all_user_snrs = [snrs[i] for snrs in snrs_list]
        plt.plot(all_user_snrs, label=f'User {i}')
        plt.ylabel(r'SNR [dB]', fontsize=20)
        plt.xlabel(r'Block Index', fontsize=20)
        plt.grid(True, which='both')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.show()
