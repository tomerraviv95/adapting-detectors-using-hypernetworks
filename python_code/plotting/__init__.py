import matplotlib as mpl

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

COLORS_DICT = {'Hypernetwork-based DeepSIC': 'red',
               'Joint DeepSIC': 'orange',
               'Online DeepSIC': 'blue'}

MARKERS_DICT = {'Hypernetwork-based DeepSIC': 'd',
                'Joint DeepSIC': 'd',
                'Online DeepSIC': 'x'}

LINESTYLES_DICT = {'Hypernetwork-based DeepSIC': 'solid',
                   'Joint DeepSIC': 'dotted',
                   'Online DeepSIC': 'solid'}
