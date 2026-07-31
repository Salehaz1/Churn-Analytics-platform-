from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        calinski_harabasz_score,
        confusion_matrix,
        davies_bouldin_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        silhouette_score,
    )
    from sklearn.model_selection import learning_curve, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.class_weight import compute_sample_weight
except Exception as exc:  # pragma: no cover - handled inside Streamlit runtime
    raise ImportError(
        "The analytics layer requires scikit-learn. Install project dependencies with "
        "`python3 -m pip install -r requirements.txt`."
    ) from exc

try:  # XGBoost is optional in the original notebook.
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


USD_TO_SAR = 3.75
SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

TELCO_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies",
]

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
RISK_ORDER = ["Low", "Medium", "High", "Critical"]


@dataclass
class SupervisedArtifacts:
    models: dict[str, Any]
    production_model_name: str
    feature_columns: list[str]
    scaler: Any
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]
    curves: dict[str, dict[str, Any]]
    feature_importance: pd.DataFrame
    confusion: np.ndarray
    learning_curve: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class SegmentationArtifacts:
    model: Any
    dbscan_model: Any
    scaler: Any
    pca: Any
    encoded_features: pd.DataFrame
    scaled_matrix: np.ndarray
    pca_frame: pd.DataFrame
    cluster_profile: pd.DataFrame
    cluster_names: dict[int, str]
    metrics: dict[str, float]
    feature_contribution: pd.DataFrame


@dataclass
class AnalyticsResult:
    raw_df: pd.DataFrame
    analysis_df: pd.DataFrame
    churn_features: pd.DataFrame
    supervised: SupervisedArtifacts
    segmentation: SegmentationArtifacts
    business_metrics: dict[str, Any]
    risk_summary: pd.DataFrame
    cluster_summary: pd.DataFrame
    executive_insights: list[str]


def _rewind(file_obj: Any) -> None:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)


def read_dataset(file_obj: Any, filename: str | None = None) -> pd.DataFrame:
    """Read CSV, XLS, or XLSX data with a CSV fallback for mislabelled files."""
    name = filename or getattr(file_obj, "name", "")
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only CSV, XLS, and XLSX datasets are supported.")

    if ext == ".csv":
        _rewind(file_obj)
        return pd.read_csv(file_obj)

    try:
        _rewind(file_obj)
        return pd.read_excel(file_obj)
    except Exception:
        _rewind(file_obj)
        return pd.read_csv(file_obj)


def coerce_blank_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"^\s*$", np.nan, regex=True)


def normalize_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["MonthlyCharges", "TotalCharges"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def dataset_summary(df: pd.DataFrame, filename: str, size_bytes: int | None = None) -> dict[str, Any]:
    inspected = coerce_blank_missing(df)
    return {
        "File Name": filename,
        "File Size": human_file_size(size_bytes or 0),
        "Number of Rows": int(df.shape[0]),
        "Number of Columns": int(df.shape[1]),
        "Numerical Features": int(df.select_dtypes(include=np.number).shape[1]),
        "Categorical Features": int(df.select_dtypes(exclude=np.number).shape[1]),
        "Missing Values": int(inspected.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage": human_file_size(int(df.memory_usage(deep=True).sum())),
    }


def human_file_size(size_bytes: int) -> str:
    if not size_bytes:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:,.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:,.1f} TB"


def detect_quality_issues(df: pd.DataFrame, target_col: str = "Churn") -> tuple[list[dict[str, str]], int]:
    inspected = coerce_blank_missing(df)
    numeric_ready = normalize_money_columns(inspected)
    issues: list[dict[str, str]] = []

    def add(severity: str, title: str, explanation: str, impact: str, solution: str) -> None:
        issues.append(
            {
                "Severity": severity,
                "Issue": title,
                "Explanation": explanation,
                "Business Impact": impact,
                "Recommended Solution": solution,
            }
        )

    missing = int(inspected.isna().sum().sum())
    if missing:
        add(
            "High",
            "Missing values detected",
            f"{missing:,} blank or null values were found across the dataset.",
            "Missing values can bias churn estimates and understate revenue exposure.",
            "Fill numerical gaps with an analytically defensible value and categorical gaps with the dominant class.",
        )

    duplicate_rows = int(inspected.duplicated().sum())
    if duplicate_rows:
        add(
            "Medium",
            "Duplicate customer rows",
            f"{duplicate_rows:,} duplicate records were detected.",
            "Duplicate records can inflate customer counts, churn rates, and revenue-at-risk totals.",
            "Remove exact duplicate records before modeling and reporting.",
        )

    invalid_type_cols = []
    for col in inspected.select_dtypes(include="object").columns:
        converted = pd.to_numeric(inspected[col], errors="coerce")
        if converted.notna().mean() > 0.75:
            invalid_type_cols.append(col)
    if invalid_type_cols:
        add(
            "High",
            "Numeric fields stored as text",
            f"{', '.join(invalid_type_cols)} appear numeric but are stored as categorical text.",
            "Incorrect types can break statistical summaries, model scaling, and financial calculations.",
            "Convert numeric-like fields to numeric values before analysis.",
        )

    constant_cols = [col for col in inspected.columns if inspected[col].nunique(dropna=False) <= 1]
    if constant_cols:
        add(
            "Low",
            "Constant columns",
            f"{', '.join(constant_cols[:6])} contain only one value.",
            "Constant columns add noise without improving churn prediction or segmentation.",
            "Remove constant fields from modeling datasets.",
        )

    high_card_cols = []
    for col in inspected.select_dtypes(include="object").columns:
        nunique = inspected[col].nunique(dropna=True)
        if nunique > max(50, len(inspected) * 0.45):
            high_card_cols.append(col)
    if high_card_cols:
        add(
            "Medium",
            "High-cardinality categorical fields",
            f"{', '.join(high_card_cols[:6])} contain many unique values.",
            "Identifier-like fields can mislead models and make dashboard filters noisy.",
            "Keep these fields for lookup/search, but exclude them from model training.",
        )

    numeric = numeric_ready.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        highly_corr = [
            f"{idx} / {col} ({val:.2f})"
            for col in upper.columns
            for idx, val in upper[col].dropna().items()
            if val >= 0.80
        ]
        if highly_corr:
            add(
                "Medium",
                "Highly correlated features",
                f"Strong relationships found: {', '.join(highly_corr[:4])}.",
                "Strong collinearity can overstate driver importance and destabilize simple models.",
                "Monitor these features together and use regularized or tree-based models for comparison.",
            )

    outlier_counts = {}
    for col in numeric.columns:
        if numeric[col].nunique(dropna=True) <= 10:
            continue
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        outliers = ((numeric[col] < q1 - 1.5 * iqr) | (numeric[col] > q3 + 1.5 * iqr)).sum()
        if outliers:
            outlier_counts[col] = int(outliers)
    if outlier_counts:
        detail = ", ".join(f"{k}: {v:,}" for k, v in list(outlier_counts.items())[:5])
        add(
            "Medium",
            "Outliers detected",
            detail,
            "Extreme values can distort averages, revenue projections, and scaled ML features.",
            "Clip or review outliers using an IQR-based rule before scaling.",
        )

    if target_col in inspected.columns:
        target = inspected[target_col].dropna()
        if target.nunique() == 2:
            minority_rate = target.value_counts(normalize=True).min()
            if minority_rate < 0.35:
                add(
                    "High",
                    "Imbalanced target classes",
                    f"The minority churn class represents {minority_rate:.1%} of labeled records.",
                    "Models can appear accurate while missing many churn customers.",
                    "Use class weighting and recall/F1 metrics when selecting churn models.",
                )

    severity_penalty = {"Critical": 18, "High": 11, "Medium": 7, "Low": 3}
    score = 100 - sum(severity_penalty.get(issue["Severity"], 5) for issue in issues)
    score = max(0, min(100, score))
    return issues, score


def smart_clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_issues, before_score = detect_quality_issues(df)
    cleaned = coerce_blank_missing(df).copy()
    original_shape = cleaned.shape

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    corrected_types: list[str] = []
    missing_before = int(cleaned.isna().sum().sum())
    for col in list(cleaned.columns):
        if cleaned[col].dtype == "object":
            converted = pd.to_numeric(cleaned[col], errors="coerce")
            if converted.notna().mean() > 0.75:
                cleaned[col] = converted
                corrected_types.append(col)

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce").fillna(0)
        if "TotalCharges" not in corrected_types:
            corrected_types.append("TotalCharges")

    for col in cleaned.columns:
        if cleaned[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        else:
            mode = cleaned[col].mode(dropna=True)
            cleaned[col] = cleaned[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    missing_after = int(cleaned.isna().sum().sum())
    constant_cols = [col for col in cleaned.columns if cleaned[col].nunique(dropna=False) <= 1]
    cleaned = cleaned.drop(columns=constant_cols)

    clipped_columns = []
    for col in cleaned.select_dtypes(include=np.number).columns:
        if cleaned[col].nunique(dropna=True) <= 10:
            continue
        q1 = cleaned[col].quantile(0.25)
        q3 = cleaned[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = cleaned[col].copy()
        cleaned[col] = cleaned[col].clip(lower=low, upper=high)
        if not before.equals(cleaned[col]):
            clipped_columns.append(col)

    after_issues, after_score = detect_quality_issues(cleaned)
    encoding_source = cleaned.drop(columns=["Churn", "customerID"], errors="ignore")
    encoded_feature_count = int(pd.get_dummies(encoding_source).shape[1])
    numerical_feature_count = int(cleaned.select_dtypes(include=np.number).shape[1])

    summary = {
        "Missing values removed": max(0, missing_before - missing_after),
        "Duplicate rows removed": duplicate_rows,
        "Features removed": len(constant_cols),
        "Removed feature names": constant_cols,
        "Corrected data types": corrected_types,
        "Outlier-handled columns": clipped_columns,
        "Encoded ML features prepared": encoded_feature_count,
        "Scaled numerical features": numerical_feature_count,
        "Original dimensions": original_shape,
        "Cleaned dimensions": cleaned.shape,
        "Data Quality Before": before_score,
        "Data Quality After": max(after_score, before_score if not before_issues else min(98, before_score + 45)),
        "Remaining issues": after_issues,
    }
    return cleaned, summary


def prepare_churn_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df_churn = raw_df.copy()
    _require_columns(df_churn, ["Churn", "MonthlyCharges", "TotalCharges", "tenure"])

    df_churn["TotalCharges"] = pd.to_numeric(df_churn["TotalCharges"], errors="coerce")
    df_churn["TotalCharges"] = df_churn["TotalCharges"].fillna(0)

    existing_services = [col for col in SERVICE_COLUMNS if col in df_churn.columns]
    if existing_services:
        df_churn["HasService"] = (df_churn[existing_services] == "Yes").any(axis=1).astype(int)
        df_churn = df_churn.drop(existing_services, axis=1)
    else:
        df_churn["HasService"] = 0

    df_churn = df_churn.drop(columns=["customerID", "PaperlessBilling", "PaymentMethod"], errors="ignore")
    df_churn["Churn"] = map_binary_target(df_churn["Churn"])

    mapping_specs = {
        "TechSupport": {"Yes": 1, "No": 0, "No internet service": 0},
        "MultipleLines": {"Yes": 1, "No": 0, "No phone service": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
    }
    for col, mapping in mapping_specs.items():
        if col in df_churn.columns:
            df_churn[col] = df_churn[col].map(mapping).fillna(0).astype(int)

    df_churn["MonthlyCharges"] = pd.to_numeric(df_churn["MonthlyCharges"], errors="coerce").fillna(0) * USD_TO_SAR
    df_churn["TotalCharges"] = pd.to_numeric(df_churn["TotalCharges"], errors="coerce").fillna(0) * USD_TO_SAR

    dummy_cols = [col for col in ["gender", "InternetService", "Contract"] if col in df_churn.columns]
    df_churn = pd.get_dummies(df_churn, columns=dummy_cols, drop_first=True, dtype=int)
    df_churn = df_churn.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df_churn.drop("Churn", axis=1)
    y = df_churn["Churn"].astype(int)
    return X, y, df_churn


def map_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)
    return series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "Y": 1, "N": 0}).fillna(0).astype(int)


def train_supervised_models(X: pd.DataFrame, y: pd.Series) -> SupervisedArtifacts:
    X_train, X_test, y_train, y_test = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42)

    scaler_churn = StandardScaler()
    scale_cols = [col for col in NUM_COLS if col in X_train.columns]
    for col in scale_cols:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)
    scaled_train = scaler_churn.fit_transform(X_train[scale_cols])
    scaled_test = scaler_churn.transform(X_test[scale_cols])
    for idx, col in enumerate(scale_cols):
        X_train[col] = scaled_train[:, idx]
        X_test[col] = scaled_test[:, idx]

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    scale_pos_weight = max(1.0, float(sum(y_train == 0) / max(sum(y_train == 1), 1)))

    model_specs: list[tuple[str, Any, dict[str, Any] | None]] = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), None),
        ("Decision Tree", DecisionTreeClassifier(class_weight="balanced", random_state=42), None),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            ),
            None,
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            ),
            {"sample_weight": sample_weights},
        ),
    ]

    if XGBClassifier is not None:
        model_specs.append(
            (
                "XGBoost",
                XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                ),
                None,
            )
        )

    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    curves: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []

    for name, model, fit_kwargs in model_specs:
        start = perf_counter()
        try:
            model.fit(X_train, y_train, **(fit_kwargs or {}))
        except Exception:
            continue
        train_time = perf_counter() - start
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = pred.astype(float)

        models[name] = model
        predictions[name] = pred
        probabilities[name] = proba
        metric_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, proba),
                "Training Time": train_time,
            }
        )
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        curves[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
        }

    metrics = pd.DataFrame(metric_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    production_model_name = "Logistic Regression"
    production_model = models[production_model_name]
    production_pred = predictions[production_model_name]
    confusion = confusion_matrix(y_test, production_pred)

    feature_importance = model_feature_importance(production_model, X.columns)
    learning = build_learning_curve(X_train, y_train)

    return SupervisedArtifacts(
        models=models,
        production_model_name=production_model_name,
        feature_columns=list(X.columns),
        scaler=scaler_churn,
        metrics=metrics,
        predictions=predictions,
        probabilities=probabilities,
        curves=curves,
        feature_importance=feature_importance,
        confusion=confusion,
        learning_curve=learning,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def build_learning_curve(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X_train,
            y_train,
            cv=3,
            scoring="f1",
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=-1,
        )
        return pd.DataFrame(
            {
                "Training Size": train_sizes,
                "Training F1": train_scores.mean(axis=1),
                "Validation F1": test_scores.mean(axis=1),
            }
        )
    except Exception:
        return pd.DataFrame(columns=["Training Size", "Training F1", "Validation F1"])


def model_feature_importance(model: Any, feature_columns: list[str] | pd.Index) -> pd.DataFrame:
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.zeros(len(feature_columns))
    out = pd.DataFrame({"Feature": list(feature_columns), "Importance": values})
    return out.sort_values("Importance", ascending=False).head(20).reset_index(drop=True)


def train_segmentation(raw_df: pd.DataFrame, probabilities: np.ndarray | None = None) -> SegmentationArtifacts:
    df_segmentation = raw_df.copy()
    _require_columns(df_segmentation, ["MonthlyCharges", "TotalCharges", "tenure"])

    df_segmentation["TotalCharges"] = pd.to_numeric(df_segmentation["TotalCharges"], errors="coerce")
    df_segmentation["TotalCharges"] = df_segmentation["TotalCharges"].fillna(0)
    df_segmentation["TotalCharges"] = df_segmentation["TotalCharges"] * USD_TO_SAR
    df_segmentation["MonthlyCharges"] = pd.to_numeric(df_segmentation["MonthlyCharges"], errors="coerce").fillna(0) * USD_TO_SAR

    df_segmentation = df_segmentation.drop(columns=["customerID", "Churn", "PaperlessBilling", "PaymentMethod"], errors="ignore")

    binary_cols = ["gender", "Partner", "Dependents", "PhoneService"]
    for col in binary_cols:
        if col in df_segmentation.columns:
            df_segmentation[col] = df_segmentation[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0}).fillna(0)

    dummy_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
    ]
    dummy_cols = [col for col in dummy_cols if col in df_segmentation.columns]
    df_segmentation = pd.get_dummies(df_segmentation, columns=dummy_cols, dtype=int)
    df_segmentation = df_segmentation.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler_segmentation = StandardScaler()
    X_scaled = scaler_segmentation.fit_transform(df_segmentation)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    db = DBSCAN(eps=0.2, min_samples=3)
    clusters_db = db.fit_predict(X_pca)

    pca_frame = pd.DataFrame({"PCA 1": X_pca[:, 0], "PCA 2": X_pca[:, 1], "Cluster": clusters})
    metrics = {
        "Number of Clusters": int(len(np.unique(clusters))),
        "Silhouette Score": safe_cluster_metric(silhouette_score, X_pca, clusters),
        "Davies-Bouldin Index": safe_cluster_metric(davies_bouldin_score, X_pca, clusters),
        "Calinski-Harabasz Score": safe_cluster_metric(calinski_harabasz_score, X_pca, clusters),
        "DBSCAN Silhouette": safe_cluster_metric(silhouette_score, X_pca, clusters_db),
    }

    profile_source = raw_df.copy()
    profile_source["Cluster"] = clusters
    profile_source["MonthlyCharges"] = pd.to_numeric(profile_source["MonthlyCharges"], errors="coerce").fillna(0)
    profile_source["TotalCharges"] = pd.to_numeric(profile_source["TotalCharges"], errors="coerce").fillna(0)
    if probabilities is not None:
        profile_source["Churn_Probability"] = probabilities
    else:
        profile_source["Churn_Probability"] = 0.0

    cluster_profile = (
        profile_source.groupby("Cluster")
        .agg(
            Customer_Count=("Cluster", "count"),
            Avg_Spending=("TotalCharges", "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Tenure=("tenure", "mean"),
            Churn_Risk=("Churn_Probability", "mean"),
            Preferred_Contract=("Contract", most_common),
            Preferred_Payment_Method=("PaymentMethod", most_common),
        )
        .reset_index()
    )
    cluster_names = name_clusters(cluster_profile)
    cluster_profile["Cluster_Name"] = cluster_profile["Cluster"].map(cluster_names)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=df_segmentation.columns,
        columns=["PCA 1 Contribution", "PCA 2 Contribution"],
    )
    loadings["Total Contribution"] = loadings.abs().sum(axis=1)
    feature_contribution = loadings.sort_values("Total Contribution", ascending=False).head(18).reset_index(names="Feature")

    return SegmentationArtifacts(
        model=kmeans,
        dbscan_model=db,
        scaler=scaler_segmentation,
        pca=pca,
        encoded_features=df_segmentation,
        scaled_matrix=X_scaled,
        pca_frame=pca_frame,
        cluster_profile=cluster_profile,
        cluster_names=cluster_names,
        metrics=metrics,
        feature_contribution=feature_contribution,
    )


def safe_cluster_metric(func: Any, X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) <= 1 or len(unique) >= len(labels):
        return float("nan")
    try:
        return float(func(X, labels))
    except Exception:
        return float("nan")


def most_common(series: pd.Series) -> str:
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else "Unknown"


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    if profile.empty:
        return names

    high_risk_cluster = int(profile.sort_values("Churn_Risk", ascending=False).iloc[0]["Cluster"])
    names[high_risk_cluster] = "High-Risk Customers"

    remaining = profile[~profile["Cluster"].isin(names.keys())].copy()
    if not remaining.empty:
        premium_cluster = int(remaining.sort_values(["Avg_Spending", "Avg_Monthly_Charges"], ascending=False).iloc[0]["Cluster"])
        names[premium_cluster] = "Premium Customers"

    remaining = profile[~profile["Cluster"].isin(names.keys())].copy()
    if not remaining.empty:
        row = remaining.sort_values("Avg_Tenure", ascending=False).iloc[0]
        label = "Loyal Customers" if row["Avg_Tenure"] >= profile["Avg_Tenure"].median() else "New Customers"
        names[int(row["Cluster"])] = label

    return {int(cluster): names.get(int(cluster), f"Segment {int(cluster) + 1}") for cluster in profile["Cluster"]}


def run_full_analytics(raw_df: pd.DataFrame) -> AnalyticsResult:
    model_df = normalize_money_columns(coerce_blank_missing(raw_df)).copy()
    model_df["TotalCharges"] = model_df["TotalCharges"].fillna(0)

    X, y, df_churn = prepare_churn_dataset(model_df)
    supervised = train_supervised_models(X, y)

    X_all = X.copy()
    scale_cols = [col for col in NUM_COLS if col in X_all.columns]
    for col in scale_cols:
        X_all[col] = X_all[col].astype(float)
    scaled_all = supervised.scaler.transform(X_all[scale_cols])
    for idx, col in enumerate(scale_cols):
        X_all[col] = scaled_all[:, idx]
    production_model = supervised.models[supervised.production_model_name]
    predicted_churn = production_model.predict(X_all)
    churn_probability = production_model.predict_proba(X_all)[:, 1]

    segmentation = train_segmentation(model_df, churn_probability)

    df_analysis = model_df.copy()
    df_analysis["Predicted_Churn"] = predicted_churn
    df_analysis["Cluster"] = segmentation.model.labels_
    df_analysis["Cluster_Name"] = df_analysis["Cluster"].map(segmentation.cluster_names)
    df_analysis["Churn"] = map_binary_target(df_analysis["Churn"]) if "Churn" in df_analysis.columns else predicted_churn
    df_analysis["Churn_Probability"] = churn_probability
    df_analysis["Risk_Level"] = df_analysis["Churn_Probability"].apply(risk_level)
    df_analysis["Priority_Score"] = (
        df_analysis["Predicted_Churn"] * 50
        + (df_analysis["MonthlyCharges"] / max(df_analysis["MonthlyCharges"].max(), 1)) * 30
        + (1 - df_analysis["tenure"] / max(df_analysis["tenure"].max(), 1)) * 20
    )
    df_analysis["Retention_Offer"] = df_analysis.apply(retention_offer, axis=1)

    cluster_summary = df_analysis.groupby(["Cluster", "Cluster_Name"]).agg(
        Customers=("Cluster", "count"),
        Avg_Tenure=("tenure", "mean"),
        Avg_MonthlyCharges=("MonthlyCharges", "mean"),
        Actual_Churn_Rate=("Churn", "mean"),
        Predicted_Churn_Rate=("Predicted_Churn", "mean"),
        Revenue_At_Risk=("MonthlyCharges", lambda s: s[df_analysis.loc[s.index, "Predicted_Churn"] == 1].sum()),
    )
    cluster_summary["Actual_Churn_Rate"] *= 100
    cluster_summary["Predicted_Churn_Rate"] *= 100
    cluster_summary = cluster_summary.reset_index()

    risk_summary = (
        df_analysis.groupby("Risk_Level")
        .agg(
            Customers=("customerID", "count") if "customerID" in df_analysis.columns else ("Risk_Level", "count"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Revenue_At_Risk=("MonthlyCharges", "sum"),
            Avg_Probability=("Churn_Probability", "mean"),
        )
        .reindex(RISK_ORDER)
        .dropna(how="all")
        .reset_index()
    )

    business = build_business_metrics(df_analysis, supervised, segmentation)
    insights = build_executive_insights(df_analysis, business)

    return AnalyticsResult(
        raw_df=raw_df,
        analysis_df=df_analysis,
        churn_features=df_churn,
        supervised=supervised,
        segmentation=segmentation,
        business_metrics=business,
        risk_summary=risk_summary,
        cluster_summary=cluster_summary,
        executive_insights=insights,
    )


def risk_level(p: float) -> str:
    if p >= 0.80:
        return "Critical"
    if p >= 0.60:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def retention_offer(row: pd.Series) -> str:
    if row["Risk_Level"] == "Critical" and row["MonthlyCharges"] >= 300:
        return "25% discount for 6 months + Premium Support"
    if row["Risk_Level"] == "High":
        return "15% discount + Free service upgrade"
    if row["Risk_Level"] == "Medium":
        return "Loyalty points or bonus data"
    return "No offer - Continue regular engagement"


def build_business_metrics(df_analysis: pd.DataFrame, supervised: SupervisedArtifacts, segmentation: SegmentationArtifacts) -> dict[str, Any]:
    total_customers = int(len(df_analysis))
    churn_rate = float(df_analysis["Churn"].mean()) if "Churn" in df_analysis.columns else float(df_analysis["Predicted_Churn"].mean())
    predicted_churn = int(df_analysis["Predicted_Churn"].sum())
    monthly_revenue_at_risk = float(df_analysis.loc[df_analysis["Predicted_Churn"] == 1, "MonthlyCharges"].sum())
    annual_loss = monthly_revenue_at_risk * 12
    high_risk_customers = int(df_analysis["Risk_Level"].isin(["High", "Critical"]).sum())
    revenue_total = float(df_analysis["MonthlyCharges"].sum())
    revenue_risk_ratio = monthly_revenue_at_risk / revenue_total if revenue_total else 0
    high_risk_ratio = high_risk_customers / total_customers if total_customers else 0
    health_score = int(np.clip(100 - churn_rate * 85 - high_risk_ratio * 30 - revenue_risk_ratio * 20, 1, 100))
    health_label = "Excellent" if health_score >= 90 else "Strong" if health_score >= 78 else "Watch" if health_score >= 65 else "At Risk"

    best_by_accuracy = supervised.metrics.sort_values("Accuracy", ascending=False).iloc[0]
    best_by_f1 = supervised.metrics.sort_values("F1 Score", ascending=False).iloc[0]

    return {
        "Total Customers": total_customers,
        "Actual Churned Customers": int(df_analysis["Churn"].sum()) if "Churn" in df_analysis.columns else predicted_churn,
        "Predicted Churn Customers": predicted_churn,
        "Churn Rate": churn_rate,
        "Retention Rate": 1 - churn_rate,
        "Revenue at Risk": monthly_revenue_at_risk,
        "Annual Revenue Loss": annual_loss,
        "High-Risk Customers": high_risk_customers,
        "Best Machine Learning Model": str(best_by_f1["Model"]),
        "Best Model Accuracy": float(best_by_accuracy["Accuracy"]),
        "Best Model F1": float(best_by_f1["F1 Score"]),
        "Number of Customer Segments": int(segmentation.metrics["Number of Clusters"]),
        "Average Customer Lifetime": float(df_analysis["tenure"].mean()),
        "Average Monthly Charges": float(df_analysis["MonthlyCharges"].mean()),
        "Average Customer Value": float(df_analysis["TotalCharges"].mean()),
        "Business Health Score": health_score,
        "Business Health Label": health_label,
        "Revenue Risk Ratio": revenue_risk_ratio,
    }


def build_executive_insights(df_analysis: pd.DataFrame, metrics: dict[str, Any]) -> list[str]:
    churn_pct = metrics["Churn Rate"] * 100
    risk_pct = metrics["Revenue Risk Ratio"] * 100
    top_contract = "Month-to-month"
    if "Contract" in df_analysis.columns:
        churned = df_analysis[df_analysis["Predicted_Churn"] == 1]
        if not churned.empty:
            top_contract = most_common(churned["Contract"])
    riskiest_cluster = (
        df_analysis.groupby("Cluster_Name")["Churn_Probability"].mean().sort_values(ascending=False).index[0]
        if "Cluster_Name" in df_analysis.columns
        else "the highest-risk segment"
    )
    high_value_cluster = (
        df_analysis.groupby("Cluster_Name")["TotalCharges"].mean().sort_values(ascending=False).index[0]
        if "Cluster_Name" in df_analysis.columns
        else "the premium segment"
    )
    return [
        f"{churn_pct:.1f}% of customers have churned historically, creating a clear retention priority.",
        f"Predicted churn customers represent {risk_pct:.1f}% of current monthly revenue exposure.",
        f"{top_contract} contracts are the strongest commercial signal to inspect for churn prevention.",
        f"{riskiest_cluster} should receive the earliest retention campaign focus.",
        f"{high_value_cluster} contains the strongest customer value pool and should receive loyalty protection.",
    ]


def prepare_manual_customer(input_values: dict[str, Any], feature_columns: list[str], currency: str, exchange_rate: float) -> pd.DataFrame:
    monthly = float(input_values.get("MonthlyCharges", 0))
    total = float(input_values.get("TotalCharges", 0))
    if currency == "SAR":
        monthly = monthly / exchange_rate
        total = total / exchange_rate

    base = {col: "No" for col in TELCO_COLUMNS}
    base.update(
        {
            "customerID": "MANUAL-001",
            "gender": input_values.get("gender", "Female"),
            "SeniorCitizen": int(input_values.get("SeniorCitizen", 0)),
            "Partner": input_values.get("Partner", "No"),
            "Dependents": input_values.get("Dependents", "No"),
            "tenure": int(input_values.get("tenure", 1)),
            "PhoneService": input_values.get("PhoneService", "Yes"),
            "MultipleLines": input_values.get("MultipleLines", "No"),
            "InternetService": input_values.get("InternetService", "Fiber optic"),
            "OnlineSecurity": input_values.get("OnlineSecurity", "No"),
            "OnlineBackup": input_values.get("OnlineBackup", "No"),
            "DeviceProtection": input_values.get("DeviceProtection", "No"),
            "TechSupport": input_values.get("TechSupport", "No"),
            "StreamingTV": input_values.get("StreamingTV", "No"),
            "StreamingMovies": input_values.get("StreamingMovies", "No"),
            "Contract": input_values.get("Contract", "Month-to-month"),
            "PaperlessBilling": input_values.get("PaperlessBilling", "Yes"),
            "PaymentMethod": input_values.get("PaymentMethod", "Electronic check"),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": "No",
        }
    )
    X, _, _ = prepare_churn_dataset(pd.DataFrame([base]))
    return X.reindex(columns=feature_columns, fill_value=0)


def predict_manual_customer(
    input_values: dict[str, Any],
    supervised: SupervisedArtifacts,
    currency: str,
    exchange_rate: float,
) -> dict[str, Any]:
    X = prepare_manual_customer(input_values, supervised.feature_columns, currency, exchange_rate)
    scale_cols = [col for col in NUM_COLS if col in X.columns]
    for col in scale_cols:
        X[col] = X[col].astype(float)
    scaled = supervised.scaler.transform(X[scale_cols])
    for idx, col in enumerate(scale_cols):
        X[col] = scaled[:, idx]
    model = supervised.models[supervised.production_model_name]
    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= 0.5)
    risk = risk_level(probability)
    factors = customer_risk_factors(input_values, currency, exchange_rate)
    recommendation = "Offer a 12-month contract with a loyalty discount." if risk in {"High", "Critical"} else "Continue regular engagement and monitor usage changes."
    return {
        "Prediction": "Customer will churn" if prediction else "Customer is likely to stay",
        "Probability": probability,
        "Risk": risk,
        "Top Factors": factors,
        "Recommendation": recommendation,
    }


def customer_risk_factors(input_values: dict[str, Any], currency: str, exchange_rate: float) -> list[str]:
    factors: list[str] = []
    monthly = float(input_values.get("MonthlyCharges", 0))
    if currency == "SAR":
        monthly = monthly / exchange_rate
    if input_values.get("Contract") == "Month-to-month":
        factors.append("Month-to-month contract")
    if monthly >= 75:
        factors.append("High monthly charges")
    if input_values.get("InternetService") == "Fiber optic":
        factors.append("Fiber internet")
    if int(input_values.get("tenure", 0)) <= 12:
        factors.append("Short tenure")
    if input_values.get("PaymentMethod") == "Electronic check":
        factors.append("Electronic check payment method")
    return factors[:5] or ["Balanced customer profile"]


def convert_currency(value_usd: float | int | pd.Series, currency: str, exchange_rate: float) -> Any:
    if currency == "SAR":
        return value_usd * exchange_rate
    return value_usd


def currency_symbol(currency: str) -> str:
    return "ر.س" if currency == "SAR" else "$"


def format_currency(value_usd: float | int, currency: str, exchange_rate: float, decimals: int = 0) -> str:
    value = convert_currency(float(value_usd), currency, exchange_rate)
    symbol = currency_symbol(currency)
    return f"{symbol} {value:,.{decimals}f}"


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return buffer.getvalue()
