import json
import os
from copy import deepcopy

from train import build_parser as build_train_parser
from train import build_run_name
from train import train_one_run
from utils import ensure_dir, save_json


SEARCH_SPACE_TYPES = {
    "features": str,
    "embed": str,
    "no_scale": int,
    "train_ratio": float,
    "val_ratio": float,
    "replace_zero_as_nan": int,
    "interpolate_missing": int,
    "seq_len": int,
    "pred_len": int,
    "forecast_horizon": int,
    "enc_in": int,
    "c_out": int,
    "d_model": int,
    "n_heads": int,
    "e_layers": int,
    "d_ff": int,
    "dropout": float,
    "activation": str,
    "top_k": int,
    "num_kernels": int,
    "use_norm": int,
    "patch_len": int,
    "stride": int,
    "gcn_true": int,
    "buildA_true": int,
    "gcn_depth": int,
    "subgraph_size": int,
    "node_dim": int,
    "dilation_exponential": int,
    "conv_channels": int,
    "residual_channels": int,
    "skip_channels": int,
    "end_channels": int,
    "layers": int,
    "propalpha": float,
    "tanhalpha": float,
    "layer_norm_affline": int,
    "txt_normalize": int,
    "train_epochs": int,
    "batch_size": int,
    "patience": int,
    "learning_rate": float,
    "lradj": str,
}

BOOLEAN_SEARCH_FIELDS = {
    "no_scale",
    "replace_zero_as_nan",
    "interpolate_missing",
    "gcn_true",
    "buildA_true",
    "layer_norm_affline",
}

OBJECTIVE_CHOICES = ("val_loss", "mae", "mse", "rmse", "mape", "mspe")


def build_parser():
    parser = build_train_parser()
    parser.description = "Optuna search for thw pipeline"
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--study_name", type=str, default="optuna_study", help="Study name")
    parser.add_argument(
        "--objective",
        type=str,
        default="val_loss",
        choices=OBJECTIVE_CHOICES,
        help="Optimization target",
    )
    parser.add_argument("--sampler_seed", type=int, default=2021, help="Seed for Optuna sampler")
    parser.add_argument(
        "--pruner_n_startup_trials",
        type=int,
        default=10,
        help="MedianPruner startup trials",
    )
    parser.add_argument(
        "--pruner_n_warmup_steps",
        type=int,
        default=3,
        help="MedianPruner warmup epochs",
    )
    parser.add_argument(
        "--pruner_interval_steps",
        type=int,
        default=1,
        help="MedianPruner interval between prune checks",
    )

    for field_name, field_type in SEARCH_SPACE_TYPES.items():
        parser.add_argument(
            f"--search-{field_name.replace('_', '-')}",
            dest=f"search_{field_name}",
            nargs="+",
            type=field_type,
            default=None,
            help=f"Candidate values for {field_name}",
        )

    return parser


def _to_python_value(field_name, value):
    if field_name in BOOLEAN_SEARCH_FIELDS:
        return bool(int(value))
    return value


def apply_search_space(trial, base_args):
    trial_args = deepcopy(base_args)
    tuned_fields = {}

    for field_name in SEARCH_SPACE_TYPES:
        search_values = getattr(base_args, f"search_{field_name}", None)
        if not search_values:
            continue

        normalized_values = [_to_python_value(field_name, value) for value in search_values]
        selected = trial.suggest_categorical(field_name, normalized_values)
        setattr(trial_args, field_name, selected)
        tuned_fields[field_name] = selected

    trial_args.model_id = f"{base_args.model_id}_trial{trial.number}"
    trial_args.results_dir = os.path.join(base_args.results_dir, base_args.study_name)
    return trial_args, tuned_fields


def load_best_val_loss(run_dir):
    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)
    if not history:
        raise ValueError(f"No training history found in {run_dir}")
    return min(row["val_loss"] for row in history)


def metric_value(metrics, objective):
    return float(metrics[objective])


def collect_search_fields(args):
    fields = []
    for field_name in SEARCH_SPACE_TYPES:
        if getattr(args, f"search_{field_name}", None):
            fields.append(field_name)
    return fields


def save_study_artifacts(study, output_dir):
    ensure_dir(output_dir)

    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    save_json(os.path.join(output_dir, "best_trial.json"), best_payload)

    trials_payload = []
    for trial in study.trials:
        trials_payload.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    save_json(os.path.join(output_dir, "trials.json"), trials_payload)


def save_param_importance_plot(study, output_dir):

    import matplotlib.pyplot as plt
    import optuna


    importances = optuna.importance.get_param_importances(study)
    if not importances:
        return

    names = list(importances.keys())
    values = list(importances.values())

    fig_height = max(4.5, 0.5 * len(names) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = list(range(len(names)))
    ax.barh(y_pos, values, color="#4C78A8")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Hyperparameter Importances")
    ax.grid(axis="x", alpha=0.25)

    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.4f}", va="center", ha="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "param_importances.png"), dpi=180)
    plt.close(fig)


def main():
    import optuna
    parser = build_parser()
    args = parser.parse_args()

    search_fields = collect_search_fields(args)


    study_output_dir = os.path.join(args.results_dir, args.study_name)
    ensure_dir(study_output_dir)
    save_json(
        os.path.join(study_output_dir, "search_config.json"),
        {
            "model": args.model,
            "objective": args.objective,
            "trials": args.trials,
            "search_fields": search_fields,
            "base_args": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("search_")
            },
            "search_space": {
                field_name: getattr(args, f"search_{field_name}")
                for field_name in search_fields
            },
        },
    )

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_n_startup_trials,
        n_warmup_steps=args.pruner_n_warmup_steps,
        interval_steps=args.pruner_interval_steps,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
    )

    def objective(trial):
        trial_args, tuned_fields = apply_search_space(trial, args)
        run_dir = os.path.join(trial_args.results_dir, build_run_name(trial_args))

        trial.set_user_attr("run_dir", run_dir)
        trial.set_user_attr("objective", args.objective)
        trial.set_user_attr("tuned_fields", tuned_fields)

        def epoch_callback(epoch_record):
            step = int(epoch_record["epoch"])
            value = float(epoch_record["val_loss"])
            trial.report(value, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch {step} with val_loss={value:.7f}"
                )

        run_dir, metrics_inverse = train_one_run(trial_args, epoch_callback=epoch_callback)

        if args.objective == "val_loss":
            objective_value = load_best_val_loss(run_dir)
        else:
            objective_value = float(metrics_inverse[args.objective])
        return objective_value

    study.optimize(objective, n_trials=args.trials)
    save_study_artifacts(study, study_output_dir)
    save_param_importance_plot(study, study_output_dir)

    print(f"Study finished: {args.study_name}")
    print(f"Best objective ({args.objective}): {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f"Saved study artifacts to: {os.path.abspath(study_output_dir)}")


if __name__ == "__main__":
    main()
