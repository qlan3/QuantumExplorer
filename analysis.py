import os
import math
from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info
from utils.helper import set_one_thread


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-10:].mean() if mode=='Train' else result['Return'][-5:].mean()
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(),
    'Return (se)': result['Return (mean)'].sem(ddof=0)
  }
  return result_dict

cfg = {
  'exp': 'exp_name',
  'merged': True,
  'x_label': 'Step',
  'y_label': 'Average Return',
  'hue_label': 'Agent',
  'show': False,
  'imgType': 'png',
  'ci': 'se',
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'lower right',
  'sweep_keys': ['actor_kwargs', 'critic_kwargs', 'qnn_layers', 'qnn_type'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  set_one_thread()
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  plotter.csv_results('Train', get_csv_result_dict, get_process_result_dict)
  # plotter.plot_results(mode='Train', indexes='all')
  
  if exp == 'sac':
    plotter.plot_indexList([1,2,3,4,5,6], 'Train', 'all')
  elif exp == 'qsac':
    plotter.plot_indexList([18,32,42,34,30], 'Train', 'top_ReUploadingVQC')
    plotter.plot_indexList([29,31,43,41,33], 'Train', 'top_NormalVQC')
    expIndexModeList = [['sac', 4, 'Train'], ['qsac', 18, 'Train'], ['qsac', 29, 'Train']]
    plotter.plot_expIndexModeList(expIndexModeList, 'sac_qsac')
    plotter.plot_indexList([4,18,32,42], 'Train', 'layers_ReUploadingVQC')
    plotter.plot_indexList([3,19,29,43], 'Train', 'layers_NormalVQC')


if __name__ == "__main__":
  exp, runs = 'sac', 10
  # exp, runs = 'qsac', 10
  analyze(exp, runs=runs)