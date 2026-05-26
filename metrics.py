import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((true - pred) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs((true - pred) / true)))


def mspe(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.square((true - pred) / true)))


def metric(pred: np.ndarray, true: np.ndarray):
    return {
        "mae": mae(pred, true),
        "mse": mse(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
        "mspe": mspe(pred, true),
    }
