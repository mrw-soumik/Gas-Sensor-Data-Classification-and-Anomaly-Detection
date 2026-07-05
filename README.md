# Gas Sensor Data Classification and Anomaly Detection

Deep learning pipeline that classifies home activities from metal-oxide gas sensor readings and flags anomalous sensor behavior with an autoencoder, implemented in both TensorFlow/Keras and PyTorch.

## Overview

This project works with time-series readings from an array of gas sensors deployed for home-activity monitoring. Two problems are addressed on the same preprocessed dataset:

- **Classification** — predict the activity associated with a sensor reading (`background`, `banana`, `wine`) using a feed-forward neural network.
- **Anomaly detection** — train an autoencoder on the sensor readings and flag samples with high reconstruction error as anomalies.

As a bonus, the classification model is re-implemented and compared across two frameworks: a Keras model tuned with `keras-tuner`, and an equivalent PyTorch model trained without tuning.

## Dataset

The data comes from the [Gas Sensors for Home Activity Monitoring Data Set](https://www.kaggle.com/datasets/gauravgawade951999/gas-sensors-for-home-activity-monitoring-data-set) (Kaggle), based on the UCI HT Sensor dataset. It ships as two files, expected at `/content/HT_Sensor_dataset.dat` and `/content/HT_Sensor_metadata.dat` when run in Google Colab (these raw `.dat` files are not included in this repository and must be downloaded separately):

- `HT_Sensor_dataset.dat` — per-timestamp readings: `id`, `time`, 8 gas sensor channels (`R1`–`R8`), `Temp.`, `Humidity`.
- `HT_Sensor_metadata.dat` — one row per session: `id`, `date`, `class` (activity label), `t0`, `dt`.

The two files are merged on `id`. After merging, standardizing, and dropping the `id`, `date`, and `class` columns, the feature matrix has **13 columns** (`time`, `R1`–`R8`, `Temp.`, `Humidity`, `t0`, `dt`) across **928,991 rows**, split 80/20 into training (743,192 rows) and test (185,799 rows) sets. There are **3 activity classes**: `background`, `banana`, `wine`.

## Repository Contents

| File | Description |
|---|---|
| `Gas_Sensor_Data_Classification_and_Anamoly_Detection.ipynb` | Main notebook containing the full pipeline described below. |
| `Gas Sensor Data Classification and Anomaly Detection - Report.pdf` | Rendered write-up of the notebook (code, explanations, and output plots) for readers who want to review the results without executing the notebook. |
| `Requirements.txt` | Package list for local setup (see note under [How to Run](#how-to-run)). |
| `LICENSE` | MIT License. |

## Notebook Structure / Reading Order

The notebook is organized into five sequential sections:

1. **Data Preprocessing** — load the two raw files, check/handle missing values, merge on `id`, standardize features with `StandardScaler`, and split into train/test sets.
2. **Exploratory Data Analysis** — summary statistics, histograms, a feature correlation heatmap, and per-activity line/box plots of sensor readings.
3. **Classification** — a Keras `Sequential` model (`Dense(64, relu) → Dropout(0.5) → Dense(32, relu) → Dropout(0.5) → Dense(3, softmax)`) trained for 20 epochs, evaluated with accuracy/precision/recall/F1, a confusion matrix, and per-class ROC curves.
4. **Anomaly Detection** — an autoencoder (`Input(13) → Dense(14, relu) → Dense(13, linear)`) trained for 20 epochs on the same features, with anomalies flagged from reconstruction error.
5. **Bonus: Hyperparameter Tuning & Framework Comparison** — `keras-tuner` random search over the classification model, and a from-scratch PyTorch reimplementation of the same architecture, compared directly.

## Methods

- **Preprocessing**: missing-value imputation (mean for numeric, mode for categorical columns in the metadata — though no missing values were actually present in either source file), `StandardScaler` normalization, 80/20 train/test split (`random_state=42`).
- **Classification model**: fully-connected network with dropout regularization, softmax output over 3 classes, Adam optimizer, categorical cross-entropy loss.
- **Anomaly detection**: a single-hidden-layer autoencoder (14-unit bottleneck) trained with MSE loss to reconstruct normalized sensor readings; samples whose reconstruction error exceeds the 95th percentile on the test set are flagged as anomalies.
- **Hyperparameter tuning**: `keras-tuner` `RandomSearch` over first/second dense layer width (32–512), dropout rate (0.0–0.5), and learning rate (1e-2, 1e-3, 1e-4), 3 trials, optimizing validation accuracy.
- **Framework comparison**: the same feed-forward architecture reimplemented in PyTorch (`nn.Linear` layers with dropout, Adam optimizer, cross-entropy loss), trained for 5 epochs without any tuning, as a baseline against the tuned Keras model.

## Results

The numbers below are taken directly from the executed notebook cell outputs (and match the accompanying PDF report).

**Classification — baseline Keras model** (20 epochs, no tuning):

| Metric | Value |
|---|---|
| Test Accuracy | 83.10% |
| Test Precision (weighted) | 83.02% |
| Test Recall (weighted) | 83.10% |
| Test F1-score (weighted) | 82.79% |

Per-class ROC AUC: `background` 0.98, `banana` 0.92, `wine` 0.91. The confusion matrix shows the `background` class is classified almost perfectly, while most errors occur between `banana` and `wine`.

**Classification — framework/tuning comparison:**

| Model | Tuning | Epochs | Test Accuracy | Test Loss |
|---|---|---|---|---|
| Keras (baseline, Section 3) | None | 20 | 83.10% | — |
| Keras + `keras-tuner` random search (Section 5) | 3 trials | 5 | 93.34% | 0.157 |
| PyTorch (same architecture) | None | 5 | 82.36% | — |

In this run, the tuned Keras model outperformed both the untuned Keras baseline and the untuned PyTorch model.

**Anomaly detection** (autoencoder):

- Final training loss: 0.0366 (MSE), final validation loss: 0.0364, after 20 epochs.
- Using the 95th-percentile reconstruction-error threshold, 9,290 of 185,799 test samples (5.00%) were flagged as anomalies. Note that this rate is a direct consequence of choosing the 95th percentile as the cutoff rather than an independent finding — a different percentile threshold would flag a different fraction by design.

## How to Run

### Google Colab (recommended, matches the notebook as written)

The notebook expects `HT_Sensor_dataset.dat` and `HT_Sensor_metadata.dat` uploaded to `/content/` in a Colab session. Most dependencies are preinstalled; install the rest with:

```python
!pip install keras-tuner
!pip install torch torchvision
```

Then run the notebook top to bottom.

### Local Environment

1. Download `HT_Sensor_dataset.dat` and `HT_Sensor_metadata.dat` from the [Kaggle dataset page](https://www.kaggle.com/datasets/gauravgawade951999/gas-sensors-for-home-activity-monitoring-data-set) and update the hardcoded `/content/...` paths in the first code cell to point to your local copies.
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner torch torchvision
   ```
   (The `Requirements.txt` file in this repository is missing `pandas` and has formatting issues; the command above lists the full set of packages the notebook actually imports.)
4. Launch the notebook:
   ```bash
   jupyter notebook Gas_Sensor_Data_Classification_and_Anamoly_Detection.ipynb
   ```

## Limitations and Notes

- The dataset's raw `.dat` files are not stored in this repository and must be obtained separately.
- The notebook's own inline commentary (and the PDF report derived from it) states different, higher comparison numbers for the tuned Keras model and PyTorch model (99.91% and 95.90% test accuracy, respectively) than what the corresponding code cells actually print (93.34% and 82.36%). This README reports only the values that appear in the executed cell outputs, since the narrative text could not be verified against them.
- The classification and hyperparameter-tuning models are trained for relatively few epochs (20 and 5 respectively) with a small random-search budget (3 trials), so the reported numbers reflect a specific run rather than an exhaustively tuned result.
- A later cell in the PyTorch section (`y_train_encoded`, `y_test_encoded`) relies on variables not defined earlier in the notebook as shown; running the notebook fresh top-to-bottom may require adding the equivalent label-encoding step before that cell.
- All plots (histograms, correlation heatmap, per-activity line/box plots, confusion matrix, ROC curves, reconstruction-error histogram) are generated inline in the notebook and reproduced in the PDF report; no standalone image files are included in the repository.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## References

1. Chollet, F. (2018). *Deep Learning with Python*. Manning Publications.
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
3. Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). TensorFlow: A System for Large-Scale Machine Learning. In *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)* (pp. 265-283).
4. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems* (pp. 8024-8035).
5. Brownlee, J. (2017). *Machine Learning Mastery with Python: Understand Your Data, Create Accurate Models, and Work Projects End-to-End*. Machine Learning Mastery.
6. Kaggle. (n.d.). *Gas Sensors for Home Activity Monitoring Data Set*. [https://www.kaggle.com/datasets/gauravgawade951999/gas-sensors-for-home-activity-monitoring-data-set](https://www.kaggle.com/datasets/gauravgawade951999/gas-sensors-for-home-activity-monitoring-data-set)
