# Infinite Recommendation Networks (∞-AE)

***This repository is a fork of the orginal Github repository and has modified/extended parts of the code***

This repository is based on the implementation of ∞-AE from the paper "Infinite Recommendation Networks: A Data-Centric Approach" [[arXiv]](https://arxiv.org/abs/2206.02626) which leverages the NTK of an infinitely-wide autoencoder for implicit-feedback recommendation. Notably, ∞-AE:

- Is easy to implement (<50 lines of relevant code)
- Has a closed-form solution
- Has only a single hyper-parameter, $\lambda$
- Even though simplistic, outperforms *all* complicated SoTA models

The paper also proposes Distill-CF: how to use ∞-AE for data distillation to create terse, high-fidelity, and synthetic data summaries for model training. We provide Distill-CF's code in a separate [GitHub repository](https://github.com/noveens/distill_cf).

If you find any module of this repository helpful for your own research, please consider citing the below paper. Thanks!

```
@article{inf_ae_distill_cf,
  title={Infinite Recommendation Networks: A Data-Centric Approach},
  author={Sachdeva, Noveen and Dhaliwal, Mehak Preet and Wu, Carole-Jean and McAuley, Julian},
  booktitle={Advances in Neural Information Processing Systems},
  series={NeurIPS '22},
  year={2022}
}
```

**Original Code Author**: Noveen Sachdeva (nosachde@ucsd.edu)

---

## Project Structure

The repository is organized as follows:

- `src/`: Contains the core source code for the Inf-AE model.
  - `extensions/`: Contains our extensions to the original codebase.
    - `baselines/`: Implementations of baseline models.
    - `dataset/`: Scripts for dataset manipulation (e.g., 3-core filtering, user sampling).
    - `fairness_diversity/`: Scripts for calculating fairness and diversity metrics.
- `ease_rec/`: Contains the implementation for the EASE model.
- `data/`: Contains the datasets.
- `configs/`: Contains configuration files for experiments and models.
- `job_scripts/`: Contains all Slurm job scripts for running experiments on a cluster.
- `slurm_out/`: Default output directory for Slurm jobs, organized by job type (inf_ae, baselines, ease, preprocessing).
- `results/`: Contains the results of experiments, such as logs and saved models.
- `log/`: Contains logs from model runs.

---

## ∞-AE Setup

#### Environment Setup

```bash
sbatch job_scripts/install_environment.job
```

#### Data Setup

This repository already includes the pre-processed data for ML-1M, Amazon Magazine, Steam, and Douban. The code used for pre-processing these datasets is in `src/preprocess.py`.

In case a user wants to pre-process a datatset themselves, follow these steps:
- Import a .inter file of the desired dataset
- Optional; In case the desired dataset is not in 3-core format:
    - Go to `job_scripts/make_3core.job` and add the name of your dataset in the final line
    - Run the command `sbatch job_scripts/make_3core.job`
- Optional; In case the .inter dataset you are working with is too large:
    - Go to `job_scripts/cut_dataset.job` and add the name of your dataset in the final line
    - Run the command `sbatch job_scripts/cut_dataset.py`
- Go to `job_scripts/get_hdf5_npz_data.job` and replace the dataset name in line 19 with the dataset of your liking (ml-1m, netflix, steam)
- Run the following command line: `sbatch job_scripts/get_hdf5_npz_data.job`
- The respective .hdf5 and .npz output file are then placed in the 'data' folder

---

#### How to train ∞-AE?

- Edit the `src/hyper_params.py` file to the dataset of your liking
- Then run the following command line: `sbatch job_scripts/run_experiment.job`
- Respective results will become visible in the 'results' folder
- If you prefer not to use a job file, type the following command to train and evaluate ∞-AE instead:

```bash
CUDA_VISIBLE_DEVICES=0 python src/main.py
```

---

## EASE Setup

#### Environment Setup

EASE makes use of the same environment as ∞-AE, which is installed using: 

```bash
sbatch job_scripts/install_environment.job
```


#### Data Setup

This repository already includes the pre-processed data for ML-1M, Amazon Magazine, Steam, and Douban. The code used for pre-processing these datasets is in `src/preprocess.py`.

In case a user wants to pre-process a datatset themselves, follow these steps:
- Import a .inter file of the desired dataset
- Optional; In case the desired dataset is not in 3-core format:
    - Go to `job_scripts/make_3core.job` and add the name of your dataset in the final line
    - Run the command `sbatch job_scripts/make_3core.job`
- Optional; In case the .inter dataset you are working with is too large:
    - Go to `job_scripts/cut_dataset.job` and add the name of your dataset in the final line
    - Run the command `sbatch job_scripts/cut_dataset.py`
- Go to `job_scripts/get_hdf5_npz_data.job` and replace the dataset name in line 19 with the dataset of your liking (ml-1m, netflix, steam)
- Run the following command line: `sbatch job_scripts/get_hdf5_npz_data.job`
- The respective .hdf5 and .npz output file are then placed in the 'data' folder

#### How to train EASE?

- Run the following command line: `sbatch job_scripts/run_ease.job`
- This job file evaluates all datasets simultaniously

---

## Multi-VAE Setup

---

---

## Light-GCN Setup

---

## Measuring Fairness/Diversity

---

## Results sneak-peak

Below are the nDCG@10 results for the datasets used in the [paper](https://arxiv.org/abs/2206.02626):

| Dataset         | PopRec | MF    | NeuMF | MVAE  | LightGCN    | EASE  | ∞-AE      |
| ----------------- | -------- | ------- | ------- | ------- | ------------- | ------- | ------------ |
| Amazon Magazine | 8.42   | 13.1  | 13.6  | 12.18 | 22.57       | 22.84 | **23.06**  |
| MovieLens-1M    | 13.84  | 25.65 | 24.44 | 22.14 | 28.85       | 29.88 | **32.82**  |
| Douban          | 11.63  | 13.21 | 13.33 | 16.17 | 16.68       | 19.48 | **24.94**  |
| Netflix         | 12.34  | 12.04 | 11.48 | 20.85 | *Timed out* | 26.83 | **30.59*** |

*Note*: ∞-AE's results on the Netflix dataset (marked with a *) are obtained by training only on 5% of the total users. Note however, all other methods are trained on the *full* dataset.

---

## MIT License
