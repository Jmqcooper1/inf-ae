

# Infinite Recommendation Networks (∞-AE)

***This repository is a fork of the orginal Github repository and has modified/extended parts of the code***

This repository is based on the implementation of ∞-AE from the paper "Infinite Recommendation Networks: A Data-Centric Approach" [[arXiv]](https://arxiv.org/abs/2206.02626) which leverages the NTK of an infinitely-wide autoencoder for implicit-feedback recommendation. Notably, ∞-AE:

- Is easy to implement (<50 lines of relevant code)
- Has a closed-form solution
- Has only a single hyper-parameter, $\lambda$
- Even though simplistic, outperforms *all* complicated SoTA models


**Original Code Author**: Noveen Sachdeva (nosachde@ucsd.edu)

---

## ∞-AE Setup

#### Environment Setup

```bash
sbatch install_environment.job
```

#### Data Setup

This repository already includes the pre-processed data for ML-1M, Amazon Magazine, Steam, and Douban. The code used for pre-processing these datasets is in `preprocess.py`.

In case a user wants to pre-process a datatset themselves, follow these steps:
- Import a .inter file of the desired dataset
- Optional; In case the desired dataset is not in 3-core format:
    - Go to make_3core.job and add the name of your dataset in the final line
    - Run the command 'sbatch make_3core.job'
- Optional; In case the .inter dataset you are working with is too large:
    - Go to cut_dataset.job and add the name of your dataset in the final line
    - Run the command 'sbatch cut_dataset.py'
- Go to get_hdf5_npz_data.job and replace the dataset name in line 19 with the dataset of your liking (ml-1m, netflix, steam)
- Run the following command line: sbatch get_hdf5_npz_data.job
- The respective .hdf5 and .npz output file are then placed in the 'data' folder

---

#### How to train ∞-AE?

- Edit the `hyper_params.py` file to the dataset of your liking
- Then run the following command line: sbatch run_experiment.job
- Respective results will become visible in the 'results' folder
- If you prefer not to use a job file, type the following command to train and evaluate ∞-AE instead:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

---

## EASE Setup

#### Environment Setup

EASE makes use of the same environment as ∞-AE, which is installed using: 

```bash
sbatch install_environment.job
```


#### Data Setup

This repository already includes the pre-processed data for ML-1M, Amazon Magazine, Steam, and Douban. The code used for pre-processing these datasets is in `preprocess.py`.

In case a user wants to pre-process a datatset themselves, follow these steps:
- Import a .inter file of the desired dataset
- Optional; In case the desired dataset is not in 3-core format:
    - Go to make_3core.job and add the name of your dataset in the final line
    - Run the command 'sbatch make_3core.job'
- Optional; In case the .inter dataset you are working with is too large:
    - Go to cut_dataset.job and add the name of your dataset in the final line
    - Run the command 'sbatch cut_dataset.py'
- Go to get_hdf5_npz_data.job and replace the dataset name in line 19 with the dataset of your liking (ml-1m, netflix, steam)
- Run the following command line: sbatch get_hdf5_npz_data.job
- The respective .hdf5 and .npz output file are then placed in the 'data' folder

#### How to train EASE?

- Run the following command line: sbatch run_ease.job
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

(!!REPLACE THIS WITH OWN RESULTS LATER)

Below are the nDCG@10 results for the datasets used in the [paper](https://arxiv.org/abs/2206.02626):

| Dataset         | PopRec | MF    | NeuMF | MVAE  | LightGCN    | EASE  | ∞-AE      |
| ----------------- | -------- | ------- | ------- | ------- | ------------- | ------- | ------------ |
| Amazon Magazine | 8.42   | 13.1  | 13.6  | 12.18 | 22.57       | 22.84 | **23.06**  |
| MovieLens-1M    | 13.84  | 25.65 | 24.44 | 22.14 | 28.85       | 29.88 | **32.82**  |
| Douban          | 11.63  | 13.21 | 13.33 | 16.17 | 16.68       | 19.48 | **24.94**  |
| Netflix         | 12.34  | 12.04 | 11.48 | 20.85 | *Timed out* | 26.83 | **30.59*** |

*Note*: ∞-AE's results on the Netflix dataset (marked with a *) are obtained by training only on 5% of the total users. Note however, all other methods are trained on the *full* dataset.

---


