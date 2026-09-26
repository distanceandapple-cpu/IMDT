# IMD-T:Enhancing Spatial Transcriptomics via Iterative Masked Diffusion Transformer guided by scRNA-seq

A novel method named stDiff investigates the potential of employing diffusion models for single-cell omics generation.

## Framework

![framework](./framework/train.png)
![framework](./framework//sample.png)

## Arguments

### IMD_T

if gene num < 512, batchsize = 2048, hiddensize = 512;   
if gene num > 512 and num < 1024, batchsize = 512, hiddensize = 1024;   
if gene num > 1024 and num < 2048, batchsize = 512, hiddensize = 2048. 

### baseline

The specific parameter settings follow those in the [Spatial Benchamark](https://github.com/QuKunLab/SpatialBenchmarking), and the baseline code is adapted from Spatial Benchmark, whose parameter settings also follow the default repositories of their respective models. \

## How to run



### datasets

[https://zenodo.org/records/22938564?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImExMzhmYWUyLWU4ZDgtNDgxNy1hODE1LWY1NTk2M2FiNTlkYyIsImRhdGEiOnt9LCJyYW5kb20iOiJiMzZjMTA4YjMyM2NkODkyYmYyZmZlNzJlZDU5OTE0YSJ9.aELZBfx9XSoTozQMuvt1GGNoT-36FqVdGO9NZ4jYiyX8VLlzVIrlea4tuhgxs-oK94-EMuIoNksnUacooaWhIA](https://zenodo.org/records/22938564?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImExMzhmYWUyLWU4ZDgtNDgxNy1hODE1LWY1NTk2M2FiNTlkYyIsImRhdGEiOnt9LCJyYW5kb20iOiJiMzZjMTA4YjMyM2NkODkyYmYyZmZlNzJlZDU5OTE0YSJ9.aELZBfx9XSoTozQMuvt1GGNoT-36FqVdGO9NZ4jYiyX8VLlzVIrlea4tuhgxs-oK94-EMuIoNksnUacooaWhIA)

### environment

```bash
conda create -n IMDT python=3.6
conda activate IMDT
pip install -r requirements.txt
```



### data preprocess

The datasets 1-9 in the experiment were all from the paper [Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type](https://www.nature.com/articles/s41592-022-01480-9). The raw data were initially processed in 'process/data_process.py' to convert the txt file to h5ad, and only the genes shared by both ST and scRNA-seq were retained. \

### run

test-IMDt for IMD_T 

```python
from model.imdt_model import DiT_IMDt
from model.imdt_scheduler import NoiseScheduler
from model.imdt_train import normal_train_IMDt
from model.sample import sample_IMDt
from process.data import *
```

```python
python test-IMDt.py --sc_data 'sc_dataset(h5ad)' --sp_data 'sp_dataset(h5ad)' --document 'IMDT_result_name' --batch_size 512 --hidden_size 1024   
```

