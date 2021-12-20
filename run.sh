export OMP_NUM_THREADS=1
parallel --eta --ungroup python main.py --config_file ./configs/sac.json --config_idx {1} ::: $(seq 1 60)
parallel --eta --ungroup python main.py --config_file ./configs/qsac.json --config_idx {1} ::: $(seq 1 480)