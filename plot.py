import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt; plt.style.use('seaborn-ticks')
from matplotlib.ticker import FuncFormatter

from utils.helper import make_dir
from utils.plotter import read_file, get_total_combination

# Set font family, bold, and font size
font = {'size': 12}
matplotlib.rc('font', **font)


class Plotter(object):
  def __init__(self, cfg):
    cfg.setdefault('ci', None)
    self.x_label = cfg['x_label']
    self.y_label = cfg['y_label']
    self.show = cfg['show']
    self.imgType = cfg['imgType']
    self.ci = cfg['ci']
    self.runs = cfg['runs']
    make_dir('./figures/')

  def get_result(self, exp, config_idx, mode):
    '''
    Given exp and config index, get the results
    '''
    total_combination = get_total_combination(exp)
    result_list = []
    for _ in range(self.runs):
      result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
      # If result file exist, read and merge
      result = read_file(result_file)
      if result is not None:
        # Add config index as a column
        result['Config Index'] = config_idx
        result_list.append(result)
      config_idx += total_combination
    ys = []
    for result in result_list:
      ys.append(result[self.y_label].to_numpy())
    # Compute x_mean, y_mean and y_ci
    ys = np.array(ys)
    x_mean = result_list[0][self.x_label].to_numpy()
    y_mean = np.mean(ys, axis=0)
    if self.ci == 'sd':
      y_ci = np.std(ys, axis=0, ddof=0)  
    elif self.ci == 'se':
      y_ci = np.std(ys, axis=0, ddof=0)/math.sqrt(len(ys))
      
    return x_mean, y_mean, y_ci


cfg = {
  'x_label': 'Step',
  'y_label': 'Average Return',
  'show': False,
  'imgType': 'png',
  'ci': 'se',
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'runs': 10
}


draw_dict = {
  'best': {
    'exp_list': ['sac', 'qsac', 'qsac'],
    'index_list': [4, 18, 29],
    'label_list': ['SAC', 'QuantumSAC (re-uploading VQC)', 'QuantumSAC (vanilla VQC)'],
    'color_list': ['tab:orange', 'tab:blue', 'tab:red'],
    'loc': 'lower right'
  },
  'layers_ReUploadingVQC': {
    'exp_list': ['qsac', 'qsac', 'qsac', 'qsac'],
    'index_list': [4, 18, 32, 42],
    'label_list': ['n=1', 'n=2', 'n=4', 'n=8'],
    'color_list': ['skyblue', 'deepskyblue', 'dodgerblue', 'tab:blue'],
    'loc': 'lower right'
  },
  'layers_VanillaVQC': {
    'exp_list': ['qsac', 'qsac', 'qsac', 'qsac'],
    'index_list': [3, 19, 29, 43],
    'label_list': ['n=1', 'n=2', 'n=4', 'n=8'],
    'color_list': ['lightcoral', 'orangered', 'red', 'darkred'],
    'loc': 'upper left'
  }
}

def learning_curve(draw_key, runs=1):
  cfg['runs'] = runs
  plotter = Plotter(cfg)
  draw_cfg = draw_dict[draw_key]

  fig, ax = plt.subplots()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  # Plot
  for i in range(len(draw_cfg['index_list'])):
    exp, config_idx, label, color = draw_cfg['exp_list'][i], draw_cfg['index_list'][i], draw_cfg['label_list'][i], draw_cfg['color_list'][i]
    x_mean, y_mean, y_ci = plotter.get_result(exp, config_idx, 'Train')
    plt.plot(x_mean, y_mean, linewidth=1.5, color=color, label=label)
    if cfg['ci'] in ['se', 'sd']:
      plt.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color, alpha=0.5)  
  # Set x and y axis
  ax.set_xlabel("Step", fontsize=12)
  ax.set_ylabel('Average Return', fontsize=12)
  plt.yticks(size=11)
  plt.xticks(size=11) 
  # # Set legend
  ax.legend(loc=draw_cfg['loc'], frameon=False, fontsize=12)
  # Adjust to show y label
  fig.subplots_adjust(left=0.16, bottom=0.12)
  # Save and show
  image_path = f'./figures/{draw_key}.{cfg["imgType"]}'
  ax.get_figure().savefig(image_path)
  if cfg['show']:
    plt.show()
  plt.clf()   # clear figure
  plt.cla()   # clear axis
  plt.close() # close window


if __name__ == "__main__":
  learning_curve('best', 10)
  learning_curve('layers_ReUploadingVQC', 10)
  learning_curve('layers_VanillaVQC', 10)