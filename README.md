# QuantumExplorer

A quantum reinforcement learning framework for exploring new ideas, based on PyTorch and PennyLane.

## Implemented algorithms

- [Variational Quantum Soft Actor-Critic (QuantumSAC)](https://arxiv.org/abs/2112.11921)

## Requirements

- Python (>=3.6)
- [PyTorch](https://pytorch.org/)
- [PennyLane](https://pennylane.readthedocs.io/en/stable/)
- [Gym && Gym Games](https://github.com/qlan3/gym-games): You may only install part of Gym (`classic_control, box2d`) by command `pip install 'gym[classic_control, box2d]'`.
- Optional: 
  - [Gym Atari](https://github.com/openai/gym/blob/master/docs/environments.md#atari)
  - [Gym Mujoco](https://github.com/openai/gym/blob/master/docs/environments.md#mujoco)
  - [PyBullet](https://pybullet.org/): `pip install pybullet`
  - [dmc2gym](https://github.com/denisyarats/dmc2gym)
- Others: Please check `requirements.txt`.


## Experiments

### Train

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including log files and the model file are saved in directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `sac.json` and configuration index `1`:

```python main.py --config_file ./configs/sac.json --config_idx 1```


### Grid Search (Optional)

First, we calculate the number of total combinations in a configuration file (e.g. `sac.json`):

`python utils/sweeper.py`

The output will be:

`Number of total combinations in sac.json: 6`

Then we run through all configuration indexes from `1` to `6`. The simplest way is a bash script:

``` bash
for index in {1..6}
do
  python main.py --config_file ./configs/sac.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule a large number of jobs:

``` bash
parallel --eta --ungroup python main.py --config_file ./configs/sac.json --config_idx {1} ::: $(seq 1 6)
```

Any configuration index that has the same remainder (divided by the number of total combinations) should has the same configuration dict. So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```
for index in 1 7 13 19 25
do
  python main.py --config_file ./configs/sac.json --config_idx $index
done
```

Or a simpler way:
```
parallel --eta --ungroup python main.py --config_file ./configs/sac.json --config_idx {1} ::: $(seq 1 6 30)
```


### Analysis (Optional)

To analysis the experimental results, just run:

`python analysis.py`

Inside `analysis.py`, `analyze` will generate a `csv` file in directory `logs/sac/0` that store all training results. More functions are available in `utils/plotter.py`.

Enjoy!


## Code of My Papers

- **Qingfeng Lan**. **Variational Quantum Soft Actor-Critic.** arXiv preprint arXiv:2112.11921, 2021. [[paper]](https://arxiv.org/abs/2112.11921)

## Cite

If you find this repo helpful to your research, you could cite my paper.

<!-- ```
@misc{Explorer,
  author = {Lan, Qingfeng},
  title = {A Quantum Reinforcement Learning Framework for Exploring New Ideas},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/qlan3/QuantumExplorer}}
}
``` -->

# Acknowledgements

- [Explorer](https://github.com/qlan3/Explorer)
- [PennyLane](https://pennylane.readthedocs.io/en/stable/)
- [TensorFlow Quantum Tutorial for Quantum Reinforcement Learning](https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning)