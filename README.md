# Classical ML Baselines for PADS

This repository contains the experiments used in the thesis
*Parkinson’s Disease Classification from Smartwatch Inertial Measurement Unit Signals
Across Structured Motor Tasks* on the **PADS** dataset. It includes both
**classical machine learning baselines** and **deep learning experiments**.
 The code supports **movement-only**, **questionnaire-only**, and **combined (stacked)** models and follows a
strict **subject-level cross-validation** protocol throughout.

The pipeline is intentionally split into **feature extraction** and **model evaluation**
stages to ensure reproducibility, avoid data leakage, and allow efficient reuse of features.

## Overview

Implemented baselines include:

- **Movement-only models** using automatically extracted **BOSS** features  
  (accelerometer, gyroscope, or all sensors)
- **Questionnaire-only models** using 30 PD non-motor symptom items
- **Stacking models** combining movement and questionnaire predictors

All experiments are evaluated at the **subject level** using stratified cross-validation,
with **balanced accuracy** as the primary metric.

## Project Structure

| Path | Description                                  |
|-----|----------------------------------------------|
| `pads/` | Root project directory                       |
| `pads/baselines/` | Classical ML baselines                       |
| `pads/baselines/cli.py` | Command-line interface                       |
| `pads/baselines/config.py` | Global paths and configuration               |
| `pads/baselines/data.py` | Dataset loading utilities                    |
| `pads/baselines/cv.py` | Subject-level CV & evaluation                |
| `pads/baselines/utils.py` | Helper functions                             |
| `pads/baselines/models/` | ML models (LR, SVM, RF, CatBoost, …)         |
| `pads/baselines/scripts/` | Utility scripts                              |
| `pads/baselines/scripts/extract_boss_features.py` | BOSS feature extraction                      |
| `pads/baselines/scripts/make_cv_splits.py` | CV split generation                          |
| `pads/baselines/boss/features_bin/` | Cached BOSS features (generated)             |
| `pads/preprocessed/` | Preprocessed data                            |
| `pads/preprocessed/file_list.csv` | Subject–file metadata                        |
| `pads/preprocessed/movement/` | Movement `.bin` files                        |
| `pads/questionnaire/` | Questionnaire JSON files                     |
| `pads/cv_splits.csv` | Subject-level CV splits (generated)          |
| `pads/notebook/DL_classification.ipynb` | Deep learning experiments (Jupyter Notebook) | 


## Requirements

- Python ≥ 3.10
- Dependencies are specified in `requirements.txt` and include:
  - `numpy`
  - `pandas`
  - `scipy`
  - `scikit-learn`
  - `pyts`
  - `joblib`
  - `catboost`
  - `lightgbm`
  - `pydantic<2` (required for `BaseSettings`)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Data Requirements
The following files and directories must exist under the project root:

* preprocessed/file_list.csv — metadata mapping files to samples

* preprocessed/movement/ — preprocessed movement .bin files

* questionnaire/ — questionnaire JSON files

No raw data preprocessing is performed automatically; all inputs are expected
to be prepared in advance.

## Usage

All commands are executed from the project root (or with the `ROOT` environment
variable set accordingly).

The CLI entry point is the package itself:

```bash
python -m baselines --help
```
1. Create cross-validation splits (only if missing)
If `cv_splits.csv` already exists in the project root, this step can be skipped.
Generate subject-level CV splits (cv_splits.csv):

```bash
python -m baselines make-splits
```
2. Extract BOSS features (only if missing)
If `baselines/boss/features_bin/` already contains the cached features, you can
skip this step.
Required for movement-based and stacked models.

```bash
python -m baselines extract-boss
```
Input:
```bash
preprocessed/movement/
```
Output:
```bash
baselines/boss/features_bin/
```
3. Run experiments with the unified run command

The run command is the central entry point.
All arguments are optional; omitted options expand to sensible defaults.
```bash
python -m baselines run [--model <MODEL>] [--task <TASK>] [--tag <TAG>] [--use-gpu]
```
* --task: pd_vs_hc or pd_vs_dd
* --model:
    * Movement/BOSS-based: lr, svm, rf, mlp, catboost, lightgbm
    * Questionnaire-only: quest_lr, quest_svm, quest_catboost
    * Stacking: stack
* --tag (movement models only): rot, acc, or all
(ignored for questionnaire-only models)
* --use-gpu: CatBoost only

Examples:
```bash
# run everything (all models × all tags × both tasks)
python -m baselines run

# run all baselines for a single task
python -m baselines run --task pd_vs_hc

# movement-only, all tags
python -m baselines run --model lr

# movement-only, single tag
python -m baselines run --model svm --task pd_vs_dd --tag acc

# CatBoost with GPU
python -m baselines run --model catboost --tag all --use-gpu

# questionnaire-only
python -m baselines run --model quest_lr
python -m baselines run --model quest_svm --task pd_vs_dd

# stacking (always uses tag --all BOSS features)
python -m baselines run --model stack
python -m baselines run --model stack --task pd_vs_hc
```
## Deep Learning Experiments (Jupyter Notebook)

In addition to the classical machine learning baselines, this repository
includes a Jupyter notebook containing the deep learning experiments used in
the thesis.

The notebook **`DL_classification.ipynb`** implements end-to-end deep learning
models operating directly on preprocessed movement signals from the PADS
dataset. The implemented architectures include:

- ARCN-based models  
- Transformer-based models  

These experiments were developed and executed primarily in **Google Colab**.
For this reason, they are provided as a standalone notebook rather than being
integrated into the command-line pipeline.

To ensure fair comparison, the notebook reuses **exactly the same**:

- preprocessed movement data  
- labels  
- subject-level cross-validation splits  

as the classical machine learning baselines.

---

### Location
The notebook is located in the directory at the project root:
```text
pads/notebook/DL_classification.ipynb
```
### Execution Environment and Paths

The notebook supports both Google Colab and local execution through a
simple path-detection mechanism:

Google Colab
If executed in Colab and the project is placed under
/content/drive/MyDrive/pads, the notebook will automatically use this
directory as the project root.

Local execution
If the repository is cloned locally and the notebook is executed from within
the project directory, the notebook will automatically treat the repository
root as the project root.

No manual path modification is required as long as the repository structure is
preserved.

### Data Requirements
The notebook expects the same directory structure as used by the classical
baselines, including:

* preprocessed/movement/ — preprocessed movement signals
* preprocessed/file_list.csv — subject and label metadata
* questionnaire/ — questionnaire data (used where applicable)
* patients/ - patients metadata (json files)

All data must already be prepared; the notebook does not perform raw data
preprocessing.

### Executing the Notebook and Reproducing Results

To reproduce the reported deep learning results:
1. Open DL_classification.ipynb
2. Execute all cells in order

The hyperparameter tuning was performed once to determine fixed model
configurations. Re-running this section is not required for reproduction and
does not affect the final reported results.

All remaining cells load the selected hyperparameters, train the models under
the predefined cross-validation protocol, and compute the final evaluation
metrics reported in the thesis.
