# Infinite Recommendation Networks (∞-AE)

***This repository is a fork of the orginal Github repository and has modified/extended parts of the code***

This repository is based on the implementation of ∞-AE from the paper "Infinite Recommendation Networks: A Data-Centric Approach" [[arXiv]](https://arxiv.org/abs/2206.02626) which leverages the NTK of an infinitely-wide autoencoder for implicit-feedback recommendation. Notably, ∞-AE:

- Is easy to implement (<50 lines of relevant code)
- Has a closed-form solution
- Has only a single hyper-parameter, $\lambda$
- Even though simplistic, outperforms *all* complicated SoTA models


**Original Code Author**: Noveen Sachdeva (nosachde@ucsd.edu), this paper uses a modified version of their codebase: [GitHub Repository: IRLab-RecSysCourse-2025/inf-ae](https://github.com/IRLab-RecSysCourse-2025/inf-ae)


---

## Project Structure

The repository is organized as follows:

- `src/`: Contains the core source code for the Inf-AE model.
  - `extensions/`: Contains our extensions to the original codebase.
    - `baselines/`: Implementations of baseline models.
    - `dataset/`: Scripts for dataset manipulation (e.g., 3-core filtering, user sampling).
    - `fairness_diversity/`: Scripts for calculating fairness and diversity metrics.
- `data/`: Contains the datasets.
- `job_scripts/`: Contains all Slurm job scripts for running experiments on a cluster.
  -   `slurm_out/`: Default output directory for Slurm jobs, organized by job type (inf_ae, baselines, ease, preprocessing).
- `results/`: Contains the results of experiments, such as logs and saved models.
- `log/`: Contains logs from model runs.

---

## ∞-AE Setup

### Data

The datasets are available as a compressed archive. To download and extract:

```bash
# Download from GitHub releases
wget https://github.com/Jmqcooper1/inf-ae/releases/download/dataset/data.tar.gz

# Extract to project root
tar -xzf data.tar.gz
```

This will create a `data/` folder containing all datasets (Netflix, Steam, ML-1M, ML-20M, Magazine, Douban).

#### Environment Setup

To install the environment, run the following command:
```bash
sbatch install_environment.job
```
remember to change the job file based on the server you are using.

#### Data Setup

This repository already includes the pre-processed data for ML-1M, Amazon Magazine, Steam, and Douban. The code used for pre-processing these datasets is in `preprocess.py`.

In case a user wants to pre-process a datatset themselves, follow these steps:
- Import a .inter file of the desired dataset
- Optional; In case the desired dataset is not in 3-core format:
    - Run the make_3core.py script in src/extensions/data
    - Command line should have the following format: python make_3core.py <dataset_name>
- Optional; In case the .inter dataset you are working with is too large:
    - Run the cut_dataset.py script in src/extensions/data
    - Command line should have the following format: python cut_dataset.py <dataset_name> [num_users]
- Run the preprocess.py script in src, command line has the following format: python preprocess.py <dataset_name>
- The respective .hdf5 and .npz output file are then placed in the 'data' folder

---

#### How to train ∞-AE?

- Edit the `hyper_params.py` file to the dataset of your liking
    - For reranking: enable one of the two reranking methods and choose to use grid search or fixed alpha
    - To compare reranking results: enable use_unfairness_gap (should always be True when using user-group)
- Then run the following command line: sbatch run_experiment.job, the job file is located in the job_scripts folder
- Respective results will become visible in the 'results' folder
- If you prefer not to use a job file, type the following command to train and evaluate ∞-AE instead:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

---

#### How to reproduce subsample size ablations for ML-20M?
- Edit the `hyper_params.py` file to the subsample size of your liking
    - Change the name to dataset name to ML-20M_XK
    - Here, X stands for number of users multiplied by 1000
    - The user can choice between 1, 5, 10, 20, 25
- Then run the following command line: sbatch run_experiment.job, the job file is located in the job_scripts folder
- Respective results will become visible in the 'results' folder
- If you prefer not to use a job file, type the following command to train and evaluate ∞-AE instead:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```


## Baseline setup

We use [RecBole](https://recbole.io/) to implement and evaluate several baseline models. The baselines are implemented in `src/extensions/baselines/` and include:

- LightGCN
- EASE
- MultiVAE

To run a baseline model or by checking `job_scripts/run_baselines.job`:

```bash
python src/extensions/baselines/run_baselines.py --model <MODEL_NAME> --dataset <DATASET_NAME> --auc
```

Where:
- `<MODEL_NAME>` can be: 'LightGCN', 'EASE', or 'MultiVAE'
- `<DATASET_NAME>` can be: 'douban', 'magazine', 'ml-1m', 'ml-20m', 'netflix', 'steam'
- `--auc` (optional flag): Computes the Area Under the ROC Curve (AUC) metric during evaluation.

To evaluate the baseline models using the generated run file, ground truth file, and train file, execute the following command or by checking `job_scripts/eval_baselines.job`:

```bash
python src/extensions/baselines/eval_baselines.py
```

---

## Results sneak-peak

Below are the nDCG@10 results for all used datasets:

| Dataset         | MultiVAE | LightGCN | EASE   | ∞-AE        |
|-----------------|----------|----------|--------|-------------|
| Amazon Magazine | 13.32    | **23.88** | 21.25  | 22.09       |
| Douban          | 18.88    | 11.69    | **28.40** | 25.18     |
| ML-1M           | 18.90    | 18.02    | 30.15  | **33.39**   |
| ML-20M          | 26.93    | 18.32    | **32.36** | 30.54     |
| Netflix         | 22.82    | 17.90    | **28.43** | 28.26     |
| Steam           | **18.90** | 28.57    | 6.29   | 1.92        |

---
## ∞-AE original paper results:
Below are the nDCG@10 results for the datasets used in the [paper](https://arxiv.org/abs/2206.02626):

| Dataset         | PopRec | MF    | NeuMF | MVAE  | LightGCN    | EASE  | ∞-AE      |
| ----------------- | -------- | ------- | ------- | ------- | ------------- | ------- | ------------ |
| Amazon Magazine | 8.42   | 13.1  | 13.6  | 12.18 | 22.57       | 22.84 | **23.06**  |
| MovieLens-1M    | 13.84  | 25.65 | 24.44 | 22.14 | 28.85       | 29.88 | **32.82**  |
| Douban          | 11.63  | 13.21 | 13.33 | 16.17 | 16.68       | 19.48 | **24.94**  |
| Netflix         | 12.34  | 12.04 | 11.48 | 20.85 | *Timed out* | 26.83 | **30.59*** |

*Note*: ∞-AE's results on the Netflix dataset (marked with a *) are obtained by training only on 5% of the total users. Note however, all other methods are trained on the *full* dataset.

---


