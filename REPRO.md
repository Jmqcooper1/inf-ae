# 🔁 Reproducibility Instructions

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.

---

## 🧱 Project Structure

```bash
.
├── data/                   # Contains raw and processed datasets
├── requirements.txt        # Python dependencies
├── README.md               # README file
├── REPRO.md                # This file
├── data.py
├── eval.py
├── model.py
├── preprocess.py
├── hyper_params.py         # Modify this file to change the hyperparamets
```

---

## ⚙️ Environment Setup


Setup project by running the following commands or by checking `job_scripts/install_enviroment.job`:

```bash
# Example -- overwrite if needed
# Create a clean environment
conda create -n inf-ae python=3.9
conda activate inf-ae

# Install JAX with CUDA support from conda-forge
conda install -c conda-forge "jaxlib=*=*cuda*" jax numpy=1.24 scipy

# Install other basic dependencies
conda install -c conda-forge matplotlib pandas

# Install project-specific requirements
pip install -r requirements.txt
```

Setup to run baselines by running the following commands or by checking `job_scripts/install_enviroment_baselines.job`:

```bash
# Example -- overwrite if needed
# Create a clean environment
conda create -n baselines python=3.9
conda activate baselines

# Install JAX with CUDA support from conda-forge
conda install -c conda-forge "jaxlib=*=*cuda*" jax

# Install RecBole library
pip install recbole

# Install Thop for Recbole 
pip install thop

# Install project-specific requirements
pip install -r requirements_baselines.txt

# Install other basic dependencies
conda install -c conda-forge matplotlib tensorflow=2.12.0 numpy=1.23.5 pandas=1.5.3 h5py=3.6.0 scipy=1.11.4
```

---

## 📂 Download & Prepare Datasets

## Creating Item Files for Gini Coefficient Analysis

- Create a tab-separated `.item` file containing at least `item_id` and `category` columns with headers
- Place datasets in the `data/` directory

## Using RecBole for Datasets

To use RecBole follow their instraction in their [website](https://recbole.io/dataset_list.html). After you downloaded you need to configure the `preprocess.py` and run it by typing 

```python
python preprocess.py
```

after that you need to just configure the `hyper_params` and you can execute the code.

### ML-1M Special Handling

For MovieLens-1M dataset, modify RecBole to handle Latin characters:

```bash
# In RecDatasets/conversion_tools/src/extended_dataset.py (line 232)
origin_data = pd.read_csv(self.item_file, delimiter=self.sep, 
                         header=None, engine='python', encoding='latin-1')
```


---

## ⚙️ Configuration

Set your parameters in the config file before training. Example:


---

## 🚀 5. Training - Evaluation

### Baselines

Training and evaluation are performed in a single step, as traditional machine learning training is not required in this approach. To run the baseline, execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

Alternatively, execute the following slurm job:

```bash
sbatch job_scripts/run_experiments.job
```


## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- [github repository](https://github.com/noveens/infinite_ae_cf)
- [paper arxiv](https://arxiv.org/abs/2206.02626)

The original authors citation:

```
@article{inf_ae_distill_cf,
  title={Infinite Recommendation Networks: A Data-Centric Approach},
  author={Sachdeva, Noveen and Dhaliwal, Mehak Preet and Wu, Carole-Jean and McAuley, Julian},
  booktitle={Advances in Neural Information Processing Systems},
  series={NeurIPS '22},
  year={2022}
}
```


