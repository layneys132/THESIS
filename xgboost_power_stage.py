import os
from copy import deepcopy

import numpy as np
import pandas as pd

from metrics import metric
from utils import ensure_dir, save_json


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost power prediction stage")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb_input_path", type=str, default=None)
    parser.add_argument("--xgb_target_source_path", type=str, default=None)
    parser.add_argument("--xgb_output_filename", type=str, default="predicted.csv")
    parser.add_argument("--xgb_date_col", type=str, default="DATE")
    parser.add_argument("--xgb_target_col", type=str, default="Target")
    parser.add_argument("--xgb_node_target_prefix", type=str, default="NSPG")
    parser.add_argument("--xgb_feature_prefixes", type=str, nargs="+", default=["HRWSPD", "HRWD_SIN", "HRWD_COS"])
    parser.add_argument("--xgb_nodes", type=int, nargs="*", default=None)
    parser.add_argument("--xgb_test_size", type=float, default=0.2)
    parser.add_argument("--xgb_valid_from_train", type=float, default=0.2)
    parser.add_argument("--xgb_n_estimators", type=int, default=3000)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.025)
    parser.add_argument("--xgb_max_depth", type=int, default=4)
    parser.add_argument("--xgb_min_child_weight", type=int, default=5)
    parser.add_argument("--xgb_subsample", type=float, default=0.85)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.85)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.05)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.5)
    parser.add_argument("--xgb_objective", type=str, default="reg:squarederror")
    parser.add_argument("--xgb_tree_method", type=str, default="hist")
    parser.add_argument("--xgb_eval_metric", type=str, default="rmse")
    parser.add_argument("--xgb_early_stopping_rounds", type=int, default=80)
    parser.add_argument("--xgb_n_jobs", type=int, default=-1)
    parser.add_argument("--xgb_verbose", type=int, default=100)
    parser.add_argument("--xgb_use_optuna", action="store_true", default=False)
    parser.add_argument("--xgb_optuna_trials", type=int, default=500)
    parser.add_argument("--xgb_optuna_show_progress", action="store_true", default=False)
    parser.add_argument("--xgb_optuna_final_objective", type=str, default="reg:absoluteerror")
    parser.add_argument("--xgb_optuna_final_eval_metric", type=str, default="mae")
    return parser


def detect_nodes(columns, target_prefix):
    nodes = []
    prefix = f"{target_prefix}_WT"
    for column in columns:
        if not str(column).startswith(prefix):
            continue
        nodes.append(int(str(column).replace(prefix, "")))

    return sorted(set(nodes))


def metric_row(name, y_true, y_pred):
    values = metric(np.asarray(y_pred, dtype=np.float32), np.asarray(y_true, dtype=np.float32))
    values["model"] = name
    return values


def make_split(frame, feature_columns, target_col, test_size, valid_from_train):
    n_rows = len(frame)
    test_start = int(n_rows * (1 - test_size)) if isinstance(test_size, float) else n_rows - int(test_size)

    train_valid = frame.iloc[:test_start].copy()
    test = frame.iloc[test_start:].copy()
    valid_start = int(len(train_valid) * (1 - valid_from_train))
    train = train_valid.iloc[:valid_start].copy()
    valid = train_valid.iloc[valid_start:].copy()

    return (
        train[feature_columns],
        train[target_col],
        valid[feature_columns],
        valid[target_col],
        test[feature_columns],
        test[target_col],
    )


def build_xgb_params(args):
    return {
        "n_estimators": args.xgb_n_estimators,
        "learning_rate": args.xgb_learning_rate,
        "max_depth": args.xgb_max_depth,
        "min_child_weight": args.xgb_min_child_weight,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample_bytree,
        "reg_alpha": args.xgb_reg_alpha,
        "reg_lambda": args.xgb_reg_lambda,
        "objective": args.xgb_objective,
        "tree_method": args.xgb_tree_method,
        "random_state": args.seed,
        "n_jobs": args.xgb_n_jobs,
        "eval_metric": args.xgb_eval_metric,
        "early_stopping_rounds": args.xgb_early_stopping_rounds,
    }


def build_optuna_trial_params(trial, args):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
        "objective": "reg:squarederror",
        "tree_method": args.xgb_tree_method,
        "random_state": args.seed,
        "n_jobs": args.xgb_n_jobs,
        "eval_metric": "rmse",
        "early_stopping_rounds": args.xgb_early_stopping_rounds,
    }


def build_optuna_final_params(best_params, args):
    final_params = dict(best_params)
    final_params.update(
        {
            "objective": args.xgb_optuna_final_objective,
            "tree_method": args.xgb_tree_method,
            "random_state": args.seed,
            "n_jobs": args.xgb_n_jobs,
            "eval_metric": args.xgb_optuna_final_eval_metric,
            "early_stopping_rounds": args.xgb_early_stopping_rounds,
        }
    )
    return final_params


def mean_absolute_error_value(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float32) - np.asarray(y_pred, dtype=np.float32))))


def save_optuna_study(study, output_dir, node):
    payload = {
        "node": node,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "trials": [
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    }
    path = os.path.join(output_dir, f"optuna_WT{node}.json")
    save_json(path, payload)
    return payload


def tune_xgboost_params(args, X_train, y_train, X_valid, y_valid, node, output_dir, XGBRegressor):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = XGBRegressor(**build_optuna_trial_params(trial, args))
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        pred = model.predict(X_valid)
        return mean_absolute_error_value(y_valid, pred)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        objective,
        n_trials=args.xgb_optuna_trials,
        show_progress_bar=args.xgb_optuna_show_progress,
    )
    study_payload = save_optuna_study(study, output_dir, node)
    return build_optuna_final_params(study.best_params, args), study_payload


def run_xgboost_power_stage(args, source_root_path, source_data_path, output_dir):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required for xgboost_power_stage") from exc

    ensure_dir(output_dir)

    input_path = os.path.join(source_root_path, args.xgb_input_path or source_data_path)


    df = pd.read_csv(input_path)
    df.columns = [str(column).strip() for column in df.columns]

    target_override_path = None
    if args.xgb_target_source_path is not None:
        target_override_path = os.path.join(source_root_path, args.xgb_target_source_path)

    if target_override_path is not None:
        target_df = pd.read_csv(target_override_path)
        df[args.xgb_target_col] = target_df[args.xgb_target_col].values

    if args.xgb_date_col in df.columns:
        df[args.xgb_date_col] = pd.to_datetime(df[args.xgb_date_col], errors="coerce", utc=True)
        df = df.sort_values(args.xgb_date_col).reset_index(drop=True)

    for column in df.columns:
        if column != args.xgb_date_col:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    nodes = args.xgb_nodes or detect_nodes(df.columns, args.xgb_node_target_prefix)


    models = {}
    metrics_rows = []
    optuna_rows = []
    output = pd.DataFrame()

    if args.xgb_date_col in df.columns:
        output[args.xgb_date_col] = df[args.xgb_date_col]

    for node in nodes:
        feature_columns = [f"{prefix}_WT{node}" for prefix in args.xgb_feature_prefixes]
        node_target = f"{args.xgb_node_target_prefix}_WT{node}"


        node_frame = df[feature_columns + [node_target]].dropna().copy()
        split = make_split(
            node_frame,
            feature_columns=feature_columns,
            target_col=node_target,
            test_size=args.xgb_test_size,
            valid_from_train=args.xgb_valid_from_train,
        )
        X_train, y_train, X_valid, y_valid, X_test, y_test = split

        if args.xgb_use_optuna:
            params, study_payload = tune_xgboost_params(
                args,
                X_train,
                y_train,
                X_valid,
                y_valid,
                node,
                output_dir,
                XGBRegressor,
            )
            optuna_rows.append(
                {
                    "node": node,
                    "best_value": study_payload["best_value"],
                    "best_params": study_payload["best_params"],
                    "best_trial_number": study_payload["best_trial_number"],
                    "final_params": params,
                }
            )
        else:
            params = build_xgb_params(args)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=args.xgb_verbose)

        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)
        full_pred = model.predict(df[feature_columns])

        models[f"WT{node}"] = model
        metrics_rows.append(metric_row(f"xgboost_valid_WT{node}", y_valid, valid_pred))
        metrics_rows.append(metric_row(f"xgboost_test_WT{node}", y_test, test_pred))

        for column in feature_columns:
            output[column] = df[column]
        output[node_target] = full_pred

    if args.xgb_target_col in df.columns:
        output[args.xgb_target_col] = df[args.xgb_target_col]

    predicted_path = os.path.join(output_dir, args.xgb_output_filename)
    output.to_csv(predicted_path, index=False)

    metrics_payload = {
        "rows": metrics_rows,
        "nodes": nodes,
        "feature_prefixes": args.xgb_feature_prefixes,
        "input_path": os.path.abspath(input_path),
        "predicted_path": os.path.abspath(predicted_path),
        "use_optuna": args.xgb_use_optuna,
        "optuna": optuna_rows,
    }
    save_json(os.path.join(output_dir, "xgboost_metrics.json"), metrics_payload)

  
    import joblib

    model_dir = os.path.join(output_dir, "models")
    ensure_dir(model_dir)
    for name, model in models.items():
        joblib.dump(model, os.path.join(model_dir, f"{name}.joblib"))


    return os.path.abspath(predicted_path), deepcopy(metrics_payload)


def main():
    args = build_parser().parse_args()
    predicted_path, _ = run_xgboost_power_stage(
        args,
        source_root_path=args.root_path,
        source_data_path=args.data_path,
        output_dir=args.output_dir,
    )
    print(f"Saved XGBoost predictions to: {predicted_path}")


if __name__ == "__main__":
    main()
