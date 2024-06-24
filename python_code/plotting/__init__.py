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
               'Joint Recurrent DeepSIC': 'orange',
               'Online Recurrent DeepSIC': 'blue',
               'Online Sequential DeepSIC': 'green',
               'Joint RNN Hypernetwork-based DeepSIC': 'black'}

MARKERS_DICT = {'Hypernetwork-based DeepSIC': 'd',
                'Joint Recurrent DeepSIC': 'd',
                'Online Recurrent DeepSIC': 'x',
                'Online Sequential DeepSIC': 'x',
                'Joint RNN Hypernetwork-based DeepSIC': 'o'}

LINESTYLES_DICT = {'Hypernetwork-based DeepSIC': 'solid',
                   'Joint Recurrent DeepSIC': 'dotted',
                   'Online Recurrent DeepSIC': 'solid',
                   'Online Sequential DeepSIC': 'dotted',
                   'Joint RNN Hypernetwork-based DeepSIC': 'solid'}
