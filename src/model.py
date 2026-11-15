
"""Machine learning model training and prediction."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.linear_model import Ridge

from src.config import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FILE,
    METADATA_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    ensure_project_directories,
)

from src.features import (
    build_supervised_dataset,
    chronological_split
)


def make_preprocessor():
    """Create preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )


def candidate_models(random_state=42):
    """Return candidate regression algorithms."""

    return {

        "Ridge Regression": Ridge(
            alpha=2.0
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    }


def calculate_metrics(actual, predicted):
    """Calculate regression evaluation metrics."""

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    return {
        "mae": float(
            mean_absolute_error(
                actual,
                predicted
            )
        ),

        "rmse": float(rmse),

        "r2": float(
            r2_score(
                actual,
                predicted
            )
        ),

        "samples": int(len(actual))
    }


def train_and_compare(
    raw_frame,
    test_fraction=0.2,
    random_state=42
):
    """Train and compare multiple regression models."""

    ensure_project_directories()

    supervised = build_supervised_dataset(
        raw_frame
    )

    if len(supervised) < 100:

        raise ValueError(
            "Insufficient supervised training examples."
        )

    train, test, cutoff = chronological_split(
        supervised,
        test_fraction
    )

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target_next_day"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target_next_day"]

    results = {}
    fitted_models = {}

    for name, estimator in candidate_models(
        random_state
    ).items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor()
                ),
                (
                    "regressor",
                    estimator
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        results[name] = calculate_metrics(
            y_test,
            predictions
        )

        fitted_models[name] = pipeline

    winner = min(
        results,
        key=lambda name: results[name]["mae"]
    )

    best_model = fitted_models[winner]

    best_predictions = best_model.predict(
        X_test
    )

    metrics = {
        "selected_model": winner,
        "cutoff_date": str(cutoff.date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "models": results,
        "evaluation_note": (
            "Chronological holdout evaluation. "
            "Future performance is not guaranteed."
        )
    }

    prediction_table = test[
        [
            "City",
            "Date",
            "target_date",
            "target_next_day"
        ]
    ].copy()

    prediction_table["predicted_aqi"] = (
        best_predictions
    )

    prediction_table["absolute_error"] = (
        prediction_table["target_next_day"] -
        prediction_table["predicted_aqi"]
    ).abs()

    joblib.dump(
        best_model,
        MODEL_FILE
    )

    metadata = {
        "selected_model": winner,
        "features": FEATURE_COLUMNS,
        "training_rows": len(train),
        "test_rows": len(test),
        "cutoff_date": str(cutoff.date()),
        "trained_at": pd.Timestamp.now().isoformat()
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    prediction_table.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    return (
        best_model,
        metrics,
        prediction_table
    )


def load_model(path=MODEL_FILE):
    """Load trained model."""

    model_path = Path(path)

    if not model_path.exists():

        raise FileNotFoundError(
            "Model not found. Run python train.py first."
        )

    return joblib.load(model_path)


def load_metadata(path=METADATA_FILE):
    """Load model metadata."""

    metadata_path = Path(path)

    if not metadata_path.exists():
        return {}

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def predict_next_day(model, feature_row):
    """Generate next-day AQI prediction."""

    prediction = float(
        model.predict(
            feature_row[FEATURE_COLUMNS]
        )[0]
    )

    return float(
        np.clip(
            prediction,
            0,
            500
        )
    )


def compare_model_table(metrics):
    """Convert model metrics into a DataFrame."""

    rows = []

    for name, values in metrics["models"].items():

        rows.append({
            "Model": name,
            "MAE": values["mae"],
            "RMSE": values["rmse"],
            "R2": values["r2"],
            "Test samples": values["samples"]
        })

    return (
        pd.DataFrame(rows)
        .sort_values("MAE")
        .reset_index(drop=True)
    )
