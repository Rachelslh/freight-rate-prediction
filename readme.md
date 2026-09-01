# Freight Rate Model

Predicts truckload **posted freight rates** from shipment details such as origin and destination, distance, equipment type, weight, date, market index, and quote signal.

The project trains an XGBoost regression model on the historical data in `data/train-test.csv`. It adds route, date, distance, interaction, frequency, and missing-value features, tunes hyperparameters with Optuna, then uses a chronological validation split to report model quality.

## Quick start

This project uses Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python train.py
```

Training configuration lives in `config.yaml`. The command writes the best Optuna parameters to `params.json` and the trained model to `xgb_final.json`.

## Data files

Place both CSV files in the `data/` folder before running the project:

- `data/train-test.csv` — historical training data, including the `posted_rate` target column.
- `data/validation.csv` — loads to score; it uses the same input columns but does not include `posted_rate`.

The default paths are configured in `config.yaml`. Use `data/validation-predictions-template.csv` as the required output format reference.

## Run inference

Train the model first so that `xgb_final.json`, `params.json` and `model_artifacts.pkl` exist, then run:

```bash
uv run python infer.py
```

This reads `data/validation.csv` and writes predictions to `validation_predictions.csv`. The output contains `load_id` and `predicted_rate` columns.

## Repository contents

- `train.py` — tunes and trains the XGBoost rate model.
- `infer.py` — runs the trained model on the validation data.
- `dataset.py` — loads data, engineers features, and makes the time-based split.
- `optimize.py` — Optuna hyperparameter search.
- `plot_features.py` — exploratory plots for the training data.
- `data/` — training data, validation inputs, and December chart inputs.
