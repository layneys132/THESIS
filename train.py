import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from dataset import create_data_loaders
from metrics import metric

from models.MTGNN import Model as MTGNNModel
from models.PatchTST import Model as PatchTSTModel
from models.STID import Model as STIDModel
from models.TimeXer import Model as TimeXerModel
from models.TimesNet import Model as TimesNetModel
from models.iTransformer import Model as ITransformerModel

from utils import EarlyStopping, adjust_learning_rate, ensure_dir, get_device, save_json, set_seed
from xgboost_power_stage import run_xgboost_power_stage

MODEL_REGISTRY = {
    "TimesNet": TimesNetModel,
    "PatchTST": PatchTSTModel,
    "iTransformer": ITransformerModel,
    "TimeXer": TimeXerModel,
    "MTGNN": MTGNNModel,
    "STIDEnsemble": STIDModel,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Forecasting pipeline")

    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--model", type=str, default="TimesNet", choices=tuple(MODEL_REGISTRY.keys()))


    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--time_col", type=str, default="date")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--no_scale", action="store_true", default=False)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--replace_zero_as_nan", action="store_true", default=False)
    parser.add_argument("--interpolate_missing", action="store_true", default=False)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--forecast_horizon", type=int, default=1)

    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--stid_hidden_dim", type=int, default=128)
    parser.add_argument("--stid_num_layers", type=int, default=6)
    parser.add_argument("--stid_num_nodes", type=int, default=0)
    parser.add_argument("--stid_in_features", type=int, default=0)
    parser.add_argument("--stid_train_noise_std", type=float, default=0.02)
    
    #MTGNN arguments
    parser.add_argument("--gcn_true", type=int, default=1)
    parser.add_argument("--buildA_true", type=int, default=1)
    parser.add_argument("--gcn_depth", type=int, default=2)
    parser.add_argument("--subgraph_size", type=int, default=20)
    
    parser.add_argument("--node_dim", type=int, default=40)
    parser.add_argument("--dilation_exponential", type=int, default=2)
    parser.add_argument("--conv_channels", type=int, default=16)
    parser.add_argument("--residual_channels", type=int, default=16)
    parser.add_argument("--skip_channels", type=int, default=32)
    parser.add_argument("--end_channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--propalpha", type=float, default=0.05)
    parser.add_argument("--tanhalpha", type=float, default=3.0)
    parser.add_argument("--layer_norm_affline", type=int, default=0)
    parser.add_argument("--txt_normalize", type=int, default=2, choices=[0, 1, 2])

    #Train arguments
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--stid_reduce_lr_patience", type=int, default=30)
    parser.add_argument("--stid_reduce_lr_factor", type=float, default=0.5)
    parser.add_argument("--stid_min_lr", type=float, default=1e-5)
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--results_dir", type=str, default="./outputs")

    #XGBoost arguments
    parser.add_argument("--use_xgboost_stage", action="store_true", default=False)
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

    #Optuna Optimizer arguments
    parser.add_argument("--xgb_use_optuna", action="store_true", default=False)
    parser.add_argument("--xgb_optuna_trials", type=int, default=500)
    parser.add_argument("--xgb_optuna_show_progress", action="store_true", default=False)
    parser.add_argument("--xgb_optuna_final_objective", type=str, default="reg:absoluteerror")
    parser.add_argument("--xgb_optuna_final_eval_metric", type=str, default="mae")

    return parser


def build_run_name(args) -> str:
    return (
        f"{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_tg{args.target}_sl{args.seq_len}_pl{args.pred_len}_fh{args.forecast_horizon}"
        f"_bs{args.batch_size}_lr{args.learning_rate}"
    )


def select_output_slice(features: str) -> int:
    return -1 if features == "MS" else 0


def metrics_to_array(metrics_dict) -> np.ndarray:
    ordered_keys = ["mae", "mse", "rmse", "mape", "mspe"]
    return np.array([metrics_dict[key] for key in ordered_keys], dtype=np.float32)


def sync_model_dimensions(args, train_dataset) -> None:
    if args.model != "STIDEnsemble":
        return

    args.stid_num_nodes = train_dataset.num_nodes
    args.stid_in_features = train_dataset.num_features
    args.enc_in = train_dataset.num_features
    args.c_out = 1


def normalize_model_output(outputs, args):
    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(-1)
    return outputs[:, -args.pred_len:, :]


def select_prediction_target(outputs, batch_y, args):
    outputs = normalize_model_output(outputs, args)
    target = batch_y
    if target.dim() == 2:
        target = target.unsqueeze(-1)

    if outputs.shape[-1] == 1 and target.shape[-1] == 1:
        return outputs, target

    feature_slice = select_output_slice(args.features)
    return outputs[:, :, feature_slice:], target[:, :, feature_slice:]


def select_array_features(array: np.ndarray, args) -> np.ndarray:
    if array.ndim != 3 or array.shape[-1] == 1:
        return array
    feature_slice = select_output_slice(args.features)
    return array[:, :, feature_slice:]


def build_optimizer(model, args):
    if args.model == "STIDEnsemble":
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.learning_rate)


def build_criterion(args):
    if args.model == "STIDEnsemble":
        return nn.HuberLoss(delta=args.huber_delta)
    return nn.MSELoss()


def build_scheduler(optimizer, args):
    if args.model != "STIDEnsemble":
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.stid_reduce_lr_factor,
        patience=args.stid_reduce_lr_patience,
        min_lr=args.stid_min_lr,
    )


def save_training_history_plot(run_dir: str, history, best_epoch=None, scale_label: str = "standardized") -> None:
    if not history:
        return
    
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val loss", linewidth=2)
    ax.plot(epochs, test_loss, label="Test loss", linewidth=2, linestyle="--")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.25, label=f"Best epoch = {best_epoch}")

    ax.set_title(f"Training History ({scale_label} loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)

    ax.legend(loc="upper right")


    output_path = os.path.join(run_dir, "training_history.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved training history plot to: {os.path.abspath(output_path)}")


def save_history_artifacts(run_dir: str, history, scale_label: str = "standardized") -> None:
    if not history:
        return

    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    save_json(os.path.join(run_dir, "history.json"), history)
    np.save(
        os.path.join(run_dir, "history.npy"),
        np.array(
            [
                [row["epoch"], row["train_loss"], row["val_loss"], row["test_loss"], row["learning_rate"]]
                for row in history
            ],
            dtype=np.float32,
        ),
    )
    save_training_history_plot(
        run_dir=run_dir,
        history=history,
        best_epoch=best_epoch,
        scale_label=scale_label,
    )


def validate(model, loader, criterion, device, args) -> float:
    losses = []
    model.eval()

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            outputs = model(batch_x, batch_x_mark)
            outputs, target = select_prediction_target(outputs, batch_y, args)

            losses.append(criterion(outputs, target).item())

    model.train()
    return float(np.mean(losses))


def train_one_run(args, epoch_callback=None):

    source_root_path = args.root_path
    source_data_path = args.data_path

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    run_name = build_run_name(args)
    run_dir = os.path.join(args.results_dir, run_name)
    ensure_dir(run_dir)
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    xgboost_stage_metadata = None

    if args.use_xgboost_stage:
        predicted_path, xgboost_stage_metadata = run_xgboost_power_stage(
            args,
            source_root_path=source_root_path,
            source_data_path=source_data_path,
            output_dir=os.path.join(run_dir, "xgboost_stage"),
        )
        args.root_path = os.path.dirname(predicted_path)
        args.data_path = os.path.basename(predicted_path)

    (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    ) = create_data_loaders(args)

    sync_model_dimensions(args, train_dataset)


    model = MODEL_REGISTRY[args.model](args).float().to(device)

    optimizer = build_optimizer(model, args)
    criterion = build_criterion(args)
    scheduler = build_scheduler(optimizer, args)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    train_steps = len(train_loader)
    time_now = time.time()
    history = []

    train_history_scale = "standardized" if (not args.no_scale) else "raw"

    try:
        for epoch in range(args.train_epochs):
            model.train()
            epoch_losses = []
            iter_count = 0
            epoch_start = time.time()

            for batch_index, (batch_x, batch_y, batch_x_mark) in enumerate(train_loader, start=1):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)

                outputs = model(batch_x, batch_x_mark)
                outputs, target = select_prediction_target(outputs, batch_y, args)
                loss = criterion(outputs, target)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                epoch_losses.append(loss.item())

                if batch_index % 100 == 0:
                    speed = (time.time() - time_now) / max(iter_count, 1)
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - batch_index)
                    print(
                        f"iters: {batch_index}, epoch: {epoch + 1} | loss: {loss.item():.7f}\n"
                        f"speed: {speed:.4f}s/iter; left time: {left_time:.4f}s"
                    )
                    iter_count = 0
                    time_now = time.time()

            train_loss = float(np.mean(epoch_losses))
            val_loss = validate(model, val_loader, criterion, device, args)
            test_loss = validate(model, test_loader, criterion, device, args)

            print(
                f"Epoch: {epoch + 1}, cost time: {time.time() - epoch_start:.2f}s | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {val_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "learning_rate": current_lr,
            }
            history.append(epoch_record)

            if epoch_callback is not None:
                epoch_callback(epoch_record)

            early_stopping(val_loss, model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if scheduler is not None:
                scheduler.step(val_loss)
            else:
                adjust_learning_rate(optimizer, epoch + 1, args)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        metrics_bundle, arrays = test(model, test_dataset, test_loader, device, args)
        feature_slice = select_output_slice(args.features)
        saved_feature_names = test_dataset.feature_names[feature_slice:]
        best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"] if history else None
        metrics_inverse = metrics_bundle["inverse"]
        metrics_scaled = metrics_bundle["scaled"]

        metadata = {
            "run_name": run_name,
            "model": args.model,
            "data": args.data,
            "data_path": os.path.abspath(os.path.join(args.root_path, args.data_path)),
            "features_mode": args.features,
            "target": args.target,
            "time_col": args.time_col,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "forecast_horizon": args.forecast_horizon,
            "feature_names": saved_feature_names,
            "raw_feature_names": test_dataset.feature_names,
            "wind_metadata": getattr(test_dataset, "wind_metadata", None),
            "xgboost_stage": xgboost_stage_metadata,
            "scaled_metrics": False,
            "scale_applied": not args.no_scale,
            "saved_arrays_are_inverse_scaled": True,
            "scaled_arrays_saved_separately": not args.no_scale,
            "metrics_are_inverse_scaled": True,
            "train_history_loss_scale": train_history_scale,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "split_borders": test_dataset.split_borders.__dict__,
            "best_epoch": best_epoch,
            "scaler_mean": test_dataset.scaler.mean.squeeze(0).tolist() if test_dataset.scaler.mean is not None else None,
            "scaler_std": test_dataset.scaler.std.squeeze(0).tolist() if test_dataset.scaler.std is not None else None,
            "metrics": metrics_inverse,
            "metrics_inverse": metrics_inverse,
            "metrics_scaled": metrics_scaled,
            "args": vars(args),
        }

        save_json(os.path.join(run_dir, "metadata.json"), metadata)
        save_json(os.path.join(run_dir, "metrics.json"), metrics_inverse)
        save_json(os.path.join(run_dir, "metrics_inverse.json"), metrics_inverse)
        save_json(os.path.join(run_dir, "metrics_scaled.json"), metrics_scaled)
        save_history_artifacts(run_dir, history, scale_label=train_history_scale)
        np.save(os.path.join(run_dir, "metrics.npy"), metrics_to_array(metrics_inverse))
        np.save(os.path.join(run_dir, "metrics_inverse.npy"), metrics_to_array(metrics_inverse))
        np.save(os.path.join(run_dir, "metrics_scaled.npy"), metrics_to_array(metrics_scaled))
        np.save(os.path.join(run_dir, "input.npy"), arrays["inputs"])
        np.save(os.path.join(run_dir, "pred.npy"), arrays["preds"])
        np.save(os.path.join(run_dir, "true.npy"), arrays["trues"])
        np.save(os.path.join(run_dir, "input_scaled.npy"), arrays["inputs_scaled"])
        np.save(os.path.join(run_dir, "pred_scaled.npy"), arrays["preds_scaled"])
        np.save(os.path.join(run_dir, "true_scaled.npy"), arrays["trues_scaled"])

        print(f"Saved run artifacts to: {os.path.abspath(run_dir)}")
        return run_dir, metrics_inverse
    except Exception:
        save_history_artifacts(run_dir, history, scale_label=train_history_scale)
        raise


def test(model, test_dataset, test_loader, device, args):
    preds_scaled = []
    trues_scaled = []
    inputs_scaled = []
    preds_inverse = []
    trues_inverse = []
    inputs_inverse = []
    scale_applied = not args.no_scale

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            outputs = model(batch_x, batch_x_mark)

            input_scaled_full = batch_x.detach().cpu().numpy()
            pred_scaled_full = normalize_model_output(outputs, args).detach().cpu().numpy()
            true_scaled_full = batch_y.detach().cpu().numpy()
            if true_scaled_full.ndim == 2:
                true_scaled_full = true_scaled_full[:, :, None]

            if scale_applied:
                input_inverse_full = test_dataset.inverse_transform(input_scaled_full)
                pred_inverse_full = test_dataset.inverse_transform(pred_scaled_full)
                true_inverse_full = test_dataset.inverse_transform(true_scaled_full)
            else:
                input_inverse_full = input_scaled_full.copy()
                pred_inverse_full = pred_scaled_full.copy()
                true_inverse_full = true_scaled_full.copy()

            inputs_scaled.append(select_array_features(input_scaled_full, args))
            preds_scaled.append(select_array_features(pred_scaled_full, args))
            trues_scaled.append(select_array_features(true_scaled_full, args))

            inputs_inverse.append(select_array_features(input_inverse_full, args))
            preds_inverse.append(select_array_features(pred_inverse_full, args))
            trues_inverse.append(select_array_features(true_inverse_full, args))

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)
    inputs_scaled = np.concatenate(inputs_scaled, axis=0)

    preds_inverse = np.concatenate(preds_inverse, axis=0)
    trues_inverse = np.concatenate(trues_inverse, axis=0)
    inputs_inverse = np.concatenate(inputs_inverse, axis=0)

    metrics_scaled = metric(preds_scaled, trues_scaled)
    metrics_inverse = metric(preds_inverse, trues_inverse)

    print(f"Test metrics (original scale): {metrics_inverse}")
    if scale_applied:
        print(f"Test metrics (standardized scale): {metrics_scaled}")

    return {
        "inverse": metrics_inverse,
        "scaled": metrics_scaled,
    }, {
        "inputs": inputs_inverse,
        "preds": preds_inverse,
        "trues": trues_inverse,
        "inputs_scaled": inputs_scaled,
        "preds_scaled": preds_scaled,
        "trues_scaled": trues_scaled,
    }


def main():

    parser = build_parser()
    args = parser.parse_args()

    train_one_run(args)


if __name__ == "__main__":
    main()
