from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analytics import RISK_ORDER, convert_currency


COLORS = {
    "blue": "#0ea5e9",
    "teal": "#14b8a6",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "violet": "#8b5cf6",
    "ink": "#0f172a",
}

RISK_COLORS = {
    "Low": "#14b8a6",
    "Medium": "#f59e0b",
    "High": "#fb7185",
    "Critical": "#ef4444",
}


def apply_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#334155"},
        margin={"l": 24, "r": 18, "t": 56 if title else 24, "b": 28},
        hoverlabel={"bgcolor": "white", "font_size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.18)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.18)", zeroline=False)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df.copy()
    for col in numeric.columns:
        if numeric[col].dtype == "object":
            converted = pd.to_numeric(numeric[col], errors="coerce")
            if converted.notna().mean() > 0.75:
                numeric[col] = converted
    corr = numeric.select_dtypes(include=np.number).corr(numeric_only=True)
    if corr.empty:
        fig = go.Figure()
        fig.add_annotation(text="No numeric features available", showarrow=False)
        return apply_layout(fig, "Correlation Heatmap")
    fig = px.imshow(
        corr,
        color_continuous_scale=["#e0f2fe", "#38bdf8", "#0f766e"],
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    return apply_layout(fig, "Correlation Heatmap")


def missing_heatmap(df: pd.DataFrame) -> go.Figure:
    sample = df.replace(r"^\s*$", np.nan, regex=True).isna().astype(int)
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)
    fig = px.imshow(
        sample.T,
        color_continuous_scale=["#ecfeff", "#ef4444"],
        aspect="auto",
        labels={"x": "Sampled Rows", "y": "Features", "color": "Missing"},
    )
    return apply_layout(fig, "Missing Value Heatmap")


def distribution_plot(df: pd.DataFrame) -> go.Figure:
    numeric_cols = list(df.select_dtypes(include=np.number).columns[:4])
    if not numeric_cols:
        fig = go.Figure()
        fig.add_annotation(text="No numeric features available", showarrow=False)
        return apply_layout(fig, "Distribution Plots")
    fig = make_subplots(rows=len(numeric_cols), cols=1, subplot_titles=numeric_cols, vertical_spacing=0.12)
    for row, col in enumerate(numeric_cols, start=1):
        fig.add_trace(go.Histogram(x=df[col], name=col, marker_color=COLORS["blue"], opacity=0.75), row=row, col=1)
    fig.update_layout(height=max(320, 190 * len(numeric_cols)), showlegend=False)
    return apply_layout(fig, "Distribution Plots")


def boxplot(df: pd.DataFrame) -> go.Figure:
    numeric_cols = list(df.select_dtypes(include=np.number).columns[:6])
    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Box(y=df[col], name=col, boxpoints="outliers", marker_color=COLORS["teal"]))
    return apply_layout(fig, "Boxplots & Outlier Detection")


def class_distribution(df: pd.DataFrame, target: str = "Churn") -> go.Figure:
    if target not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No target column found", showarrow=False)
        return apply_layout(fig, "Class Distribution")
    counts = df[target].value_counts().reset_index()
    counts.columns = [target, "Customers"]
    fig = px.bar(counts, x=target, y="Customers", color=target, color_discrete_sequence=[COLORS["teal"], COLORS["red"]])
    return apply_layout(fig, "Target / Class Distribution")


def risk_distribution(df: pd.DataFrame, donut: bool = True) -> go.Figure:
    counts = df["Risk_Level"].value_counts().reindex(RISK_ORDER).dropna().reset_index()
    counts.columns = ["Risk Level", "Customers"]
    fig = px.pie(
        counts,
        names="Risk Level",
        values="Customers",
        hole=0.58 if donut else 0,
        color="Risk Level",
        color_discrete_map=RISK_COLORS,
    )
    return apply_layout(fig, "Customer Risk Distribution")


def revenue_by_risk(df: pd.DataFrame, currency: str, exchange_rate: float) -> go.Figure:
    work = df.copy()
    work["Revenue"] = convert_currency(work["MonthlyCharges"], currency, exchange_rate)
    summary = work.groupby("Risk_Level", as_index=False)["Revenue"].sum()
    summary["Risk_Level"] = pd.Categorical(summary["Risk_Level"], RISK_ORDER, ordered=True)
    summary = summary.sort_values("Risk_Level")
    fig = px.bar(
        summary,
        x="Risk_Level",
        y="Revenue",
        color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        labels={"Risk_Level": "Risk Level"},
    )
    return apply_layout(fig, "Revenue by Risk Level")


def revenue_by_segment(df: pd.DataFrame, currency: str, exchange_rate: float) -> go.Figure:
    work = df.copy()
    work["Revenue"] = convert_currency(work["MonthlyCharges"], currency, exchange_rate)
    summary = work.groupby("Cluster_Name", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    fig = px.bar(summary, x="Cluster_Name", y="Revenue", color="Cluster_Name", color_discrete_sequence=px.colors.qualitative.Set2)
    return apply_layout(fig, "Revenue by Customer Segment")


def revenue_distribution(df: pd.DataFrame, currency: str, exchange_rate: float) -> go.Figure:
    values = convert_currency(df["MonthlyCharges"], currency, exchange_rate)
    fig = px.histogram(x=values, nbins=36, labels={"x": f"Monthly Charges ({currency})", "y": "Customers"})
    fig.update_traces(marker_color=COLORS["blue"], opacity=0.78)
    return apply_layout(fig, "Revenue Distribution")


def revenue_trend(df: pd.DataFrame, currency: str, exchange_rate: float) -> go.Figure:
    work = df.copy()
    work["MonthlyRevenue"] = convert_currency(work["MonthlyCharges"], currency, exchange_rate)
    work["Tenure Bucket"] = pd.cut(work["tenure"], bins=[-1, 6, 12, 24, 36, 48, 60, 72], labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
    summary = work.groupby("Tenure Bucket", observed=False)["MonthlyRevenue"].sum().reset_index()
    fig = px.line(summary, x="Tenure Bucket", y="MonthlyRevenue", markers=True)
    fig.update_traces(line_color=COLORS["teal"], marker_size=9)
    return apply_layout(fig, "Revenue Trend by Customer Tenure")


def clv_by_segment(df: pd.DataFrame, currency: str, exchange_rate: float) -> go.Figure:
    work = df.copy()
    work["CLV"] = convert_currency(work["TotalCharges"], currency, exchange_rate)
    summary = work.groupby("Cluster_Name", as_index=False)["CLV"].mean().sort_values("CLV", ascending=False)
    fig = px.bar(summary, x="Cluster_Name", y="CLV", color="Cluster_Name", color_discrete_sequence=px.colors.qualitative.Pastel)
    return apply_layout(fig, "Customer Lifetime Value by Segment")


def confusion_matrix_fig(matrix: np.ndarray) -> go.Figure:
    fig = px.imshow(
        matrix,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Customers"},
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"],
        color_continuous_scale=["#ecfeff", "#0ea5e9", "#0f766e"],
    )
    return apply_layout(fig, "Confusion Matrix")


def roc_curve_fig(curves: dict[str, dict[str, np.ndarray]]) -> go.Figure:
    fig = go.Figure()
    for name, data in curves.items():
        fig.add_trace(go.Scatter(x=data["fpr"], y=data["tpr"], mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line={"dash": "dash", "color": "#94a3b8"}))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return apply_layout(fig, "ROC Curve")


def precision_recall_fig(curves: dict[str, dict[str, np.ndarray]]) -> go.Figure:
    fig = go.Figure()
    for name, data in curves.items():
        fig.add_trace(go.Scatter(x=data["recall"], y=data["precision"], mode="lines", name=name))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return apply_layout(fig, "Precision-Recall Curve")


def learning_curve_fig(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not data.empty:
        fig.add_trace(go.Scatter(x=data["Training Size"], y=data["Training F1"], mode="lines+markers", name="Training F1"))
        fig.add_trace(go.Scatter(x=data["Training Size"], y=data["Validation F1"], mode="lines+markers", name="Validation F1"))
    fig.update_yaxes(range=[0, 1], title="F1 Score")
    fig.update_xaxes(title="Training Samples")
    return apply_layout(fig, "Learning Curve")


def feature_importance_fig(data: pd.DataFrame, title: str = "Feature Importance") -> go.Figure:
    plot = data.sort_values(data.columns[-1], ascending=True).tail(15)
    value_col = "Importance" if "Importance" in plot.columns else plot.columns[-1]
    fig = px.bar(plot, x=value_col, y="Feature", orientation="h", color=value_col, color_continuous_scale=["#bae6fd", "#14b8a6"])
    return apply_layout(fig, title)


def prediction_distribution(probabilities: np.ndarray) -> go.Figure:
    fig = px.histogram(x=probabilities, nbins=30, labels={"x": "Churn Probability", "y": "Customers"})
    fig.update_traces(marker_color=COLORS["violet"], opacity=0.78)
    return apply_layout(fig, "Prediction Distribution")


def actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    table = pd.crosstab(pd.Series(y_true, name="Actual"), pd.Series(y_pred, name="Predicted")).reset_index().melt(id_vars="Actual", var_name="Predicted", value_name="Customers")
    fig = px.bar(table, x="Actual", y="Customers", color="Predicted", barmode="group", color_discrete_sequence=[COLORS["teal"], COLORS["red"]])
    return apply_layout(fig, "Actual vs Predicted")


def pca_projection(pca_frame: pd.DataFrame) -> go.Figure:
    work = pca_frame.copy()
    work["Cluster"] = work["Cluster"].astype(str)
    fig = px.scatter(work, x="PCA 1", y="PCA 2", color="Cluster", opacity=0.75, color_discrete_sequence=px.colors.qualitative.Set2)
    return apply_layout(fig, "PCA Projection")


def cluster_size_distribution(df: pd.DataFrame) -> go.Figure:
    summary = df["Cluster_Name"].value_counts().reset_index()
    summary.columns = ["Cluster", "Customers"]
    fig = px.bar(summary, x="Cluster", y="Customers", color="Cluster", color_discrete_sequence=px.colors.qualitative.Set2)
    return apply_layout(fig, "Cluster Size Distribution")


def cluster_heatmap(profile: pd.DataFrame, currency: str, exchange_rate: float) -> go.Figure:
    if profile.empty:
        return apply_layout(go.Figure(), "Cluster Heatmap")
    work = profile.copy()
    work["Avg_Spending"] = convert_currency(work["Avg_Spending"], currency, exchange_rate)
    work["Avg_Monthly_Charges"] = convert_currency(work["Avg_Monthly_Charges"], currency, exchange_rate)
    metrics = work[["Cluster_Name", "Customer_Count", "Avg_Spending", "Avg_Monthly_Charges", "Avg_Tenure", "Churn_Risk"]].set_index("Cluster_Name")
    normalized = (metrics - metrics.min()) / (metrics.max() - metrics.min()).replace(0, 1)
    fig = px.imshow(normalized, text_auto=".2f", color_continuous_scale=["#ecfeff", "#38bdf8", "#0f766e"], aspect="auto")
    return apply_layout(fig, "Cluster Heatmap")


def cluster_radar(profile: pd.DataFrame, currency: str, exchange_rate: float) -> go.Figure:
    if profile.empty:
        return apply_layout(go.Figure(), "Cluster Radar Chart")
    work = profile.copy()
    work["Avg_Spending"] = convert_currency(work["Avg_Spending"], currency, exchange_rate)
    work["Avg_Monthly_Charges"] = convert_currency(work["Avg_Monthly_Charges"], currency, exchange_rate)
    cols = ["Avg_Spending", "Avg_Monthly_Charges", "Avg_Tenure", "Churn_Risk"]
    normalized = work[cols].copy()
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min()).replace(0, 1)
    fig = go.Figure()
    categories = ["Avg Spending", "Monthly Charges", "Tenure", "Churn Risk"]
    for idx, row in normalized.iterrows():
        values = list(row.values) + [row.values[0]]
        fig.add_trace(go.Scatterpolar(r=values, theta=categories + [categories[0]], fill="toself", name=work.loc[idx, "Cluster_Name"]))
    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}}, showlegend=True)
    return apply_layout(fig, "Radar Chart")
