import json

import optuna
from sklearn.metrics import mean_squared_error
import xgboost as xgb


class Optimizer:
    def __init__(self, config):
        self.n_trials = config.N_TRIALS
        self.seed = config.RANDOM_SEED
        self.out_path = config.params_out_file
        self.cfg = config
            
    def __call__(self, X_train, y_train_log, X_val, y_val_log):
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": self.cfg.n_estimators,
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "early_stopping_rounds": self.cfg.early_stopping_rounds,
                "random_state": self.seed,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
            preds = model.predict(X_val)
            return mean_squared_error(y_val_log, preds)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials)

        print("Best params:", study.best_params)
        print("Best log-RMSE:", study.best_value ** 0.5)
        
        with open(self.out_path, "w") as f:
            json.dump(study.best_params, f, indent=2)
            
        return study.best_params