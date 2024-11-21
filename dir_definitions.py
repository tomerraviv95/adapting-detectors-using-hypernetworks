import os

# main folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'python_code')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')

# subfolders
CONFIG_PATH = os.path.join(CODE_DIR, 'config.yaml')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
COST2100_TRAIN_DIR = os.path.join(RESOURCES_DIR, 'cost2100_train')
COST2100_TEST_DIR = os.path.join(RESOURCES_DIR, 'cost2100_test')
QUADRIGA_TRAIN_DIR = os.path.join(RESOURCES_DIR, 'quadriga_train')
QUADRIGA_TEST_DIR = os.path.join(RESOURCES_DIR, 'quadriga_test')