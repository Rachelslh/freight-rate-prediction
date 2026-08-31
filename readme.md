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

## Repository contents

- `train.py` — tunes and trains the XGBoost rate model.
- `dataset.py` — loads data, engineers features, and makes the time-based split.
- `optimize.py` — Optuna hyperparameter search.
- `plot_features.py` — exploratory plots for the training data.
- `data/` — historical training data, validation inputs, and December chart inputs.

