"""Train a bidirectional LSTM flood classifier on cleaned real data."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = DATA_DIR / "india_flood_clean.csv"

FEATURE_COLUMNS = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "water_level_m",
    "river_discharge_m3_s",
    "elevation_m",
    "population_density",
    "year",
    "month",
    "day_of_year",
    "is_monsoon",
    "rainfall_7day",
    "rainfall_30day",
    "api",
    "river_rise_rate",
    "month_sin",
    "month_cos",
    "rainfall_intensity",
    "rainfall_river_interaction",
    "discharge_per_water_level",
    "terrain_rain_risk",
]
TARGET_COLUMN = "flood_occurred"
RANDOM_STATE = 42
SEQ_LEN = 14
BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 8
LR = 0.001
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.05), 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FloodSequenceDataset(Dataset):
    """Torch dataset for fixed-length flood feature sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.labels[idx])


class FloodLSTM(nn.Module):
    """Bidirectional LSTM binary classifier."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.30,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 96),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.head(hidden).squeeze(1)


def load_real_data() -> tuple[np.ndarray, np.ndarray]:
    """Load cleaned real data and return numeric features and target."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Clean real data not found: {CLEAN_DATA_PATH}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {CLEAN_DATA_PATH}: {missing}")

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0).to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    print(f"Loaded real clean data: {len(df):,} rows")
    print(f"Overall flood rate: {y.mean():.2%}")
    return X, y


def stratified_row_split(X: np.ndarray, y: np.ndarray):
    """Create the requested stratified 70/15/15 row split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    print("\nStratified row splits:")
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:5}: {len(labels):,} rows | flood rate {labels.mean():.2%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sequences(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create 14-step sequences from a split, labeling each sequence by its last row."""
    if len(X) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} rows to create LSTM sequences.")
    X_seq = np.asarray([X[i : i + SEQ_LEN] for i in range(len(X) - SEQ_LEN + 1)], dtype=np.float32)
    y_seq = np.asarray([y[i + SEQ_LEN - 1] for i in range(len(X) - SEQ_LEN + 1)], dtype=np.int64)
    return X_seq, y_seq


def create_split_sequences(X_train, X_val, X_test, y_train, y_val, y_test):
    """Create sequences after stratified row splitting and report sequence rates."""
    X_train_seq, y_train_seq = create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test)
    print("\nLSTM sequences:")
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:5}: {len(labels):,} rows before sequencing | flood rate {labels.mean():.2%}")
    for name, labels in [("Train", y_train_seq), ("Val", y_val_seq), ("Test", y_test_seq)]:
        print(f"  {name:5}: {len(labels):,} sequences | flood rate {labels.mean():.2%}")
    return X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq


def scale_sequences(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """Scale sequence features with a scaler fit only on training timesteps."""
    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    joblib.dump(scaler, MODELS_DIR / "lstm_real_scaler.pkl")
    return X_train_scaled, X_val_scaled, X_test_scaled


def make_train_loader(X_train: np.ndarray, y_train: np.ndarray) -> DataLoader:
    """Create a weighted sampler so flood sequences are seen more often."""
    labels = y_train.astype(int)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        FloodSequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )


def predict_probabilities(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities and labels for a loader."""
    model.eval()
    probs: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            labels.extend(y_batch.numpy().reshape(-1))
    return np.asarray(probs), np.asarray(labels).astype(int)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Pick the validation threshold with the best F1 score."""
    best_threshold = 0.50
    best_f1 = -1.0
    print("\nThreshold tuning:")
    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        print(f"  {threshold:.2f}: F1={score:.4f}")
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    print(f"Best threshold: {best_threshold:.2f} | val F1={best_f1:.4f}")
    return best_threshold, best_f1


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute final binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Best Threshold": threshold,
    }


def train_model() -> tuple[FloodLSTM, dict[str, float]]:
    """Train, evaluate, and save the real-data LSTM model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X, y = load_real_data()
    X_train_rows, X_val_rows, X_test_rows, y_train_rows, y_val_rows, y_test_rows = stratified_row_split(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = create_split_sequences(
        X_train_rows,
        X_val_rows,
        X_test_rows,
        y_train_rows,
        y_val_rows,
        y_test_rows,
    )
    X_train, X_val, X_test = scale_sequences(X_train, X_val, X_test)

    train_loader = make_train_loader(X_train, y_train)
    val_loader = DataLoader(FloodSequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FloodSequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = FloodLSTM(input_size=X_train.shape[-1]).to(DEVICE)
    no_flood = int((y_train == 0).sum())
    flood = int((y_train == 1).sum())
    pos_weight = torch.tensor([no_flood / max(flood, 1)], dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    print(f"\nTraining LSTM on {DEVICE} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_f1 = -1.0
    best_threshold = 0.50
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        val_prob, val_true = predict_probabilities(model, val_loader)
        epoch_threshold, epoch_f1 = tune_threshold(val_true, val_prob)
        val_auc = roc_auc_score(val_true, val_prob)
        avg_loss = total_loss / len(y_train)
        scheduler.step(epoch_f1)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={avg_loss:.4f} | "
            f"val_f1={epoch_f1:.4f} | val_auc={val_auc:.4f}"
        )

        if epoch_f1 > best_val_f1:
            best_val_f1 = epoch_f1
            best_threshold = epoch_threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_prob, test_true = predict_probabilities(model, test_loader)
    metrics = metric_dict(test_true, test_prob, best_threshold)

    print("\n" + "=" * 72)
    print("LSTM REAL DATA RESULTS")
    print("=" * 72)
    for name, value in metrics.items():
        print(f"{name:>14}: {value:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            test_true,
            (test_prob >= best_threshold).astype(int),
            target_names=["No Flood", "Flood"],
            zero_division=0,
        )
    )

    model_path = MODELS_DIR / "lstm_real_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": SEQ_LEN,
            "best_threshold": best_threshold,
            "metrics": metrics,
        },
        model_path,
    )
    joblib.dump(metrics, MODELS_DIR / "lstm_real_metrics.pkl")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "lstm_real_feature_names.pkl")
    print(f"Saved LSTM model to {model_path}")
    return model, metrics


if __name__ == "__main__":
    train_model()
