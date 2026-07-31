from __future__ import annotations

from io import BytesIO
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .analytics import AnalyticsResult, dataframe_to_excel_bytes, format_currency, format_percent


def build_executive_pdf(result: AnalyticsResult, currency: str, exchange_rate: float) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "PremiumTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=10,
        spaceAfter=8,
    )
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13, textColor=colors.HexColor("#334155"))

    metrics = result.business_metrics
    rows = [
        ["Metric", "Value"],
        ["Total Customers", f"{metrics['Total Customers']:,}"],
        ["Churn Rate", format_percent(metrics["Churn Rate"])],
        ["Retention Rate", format_percent(metrics["Retention Rate"])],
        ["Revenue at Risk", format_currency(metrics["Revenue at Risk"], currency, exchange_rate)],
        ["Annual Revenue Loss", format_currency(metrics["Annual Revenue Loss"], currency, exchange_rate)],
        ["High-Risk Customers", f"{metrics['High-Risk Customers']:,}"],
        ["Best ML Model", metrics["Best Machine Learning Model"]],
        ["Best Model Accuracy", format_percent(metrics["Best Model Accuracy"])],
        ["Customer Segments", f"{metrics['Number of Customer Segments']:,}"],
        ["Business Health", f"{metrics['Business Health Score']} / 100 - {metrics['Business Health Label']}"],
    ]

    story: list[Any] = [
        Paragraph("Customer Churn Analytics Platform", title),
        Paragraph("Executive Report", h2),
        Paragraph("This report summarizes churn exposure, customer risk, model performance, and recommended retention focus areas.", body),
        Spacer(1, 10),
        _styled_table(rows, col_widths=[2.35 * inch, 3.9 * inch]),
        Paragraph("Executive AI Summary", h2),
    ]
    for insight in result.executive_insights:
        story.append(Paragraph(f"- {insight}", body))
        story.append(Spacer(1, 3))

    model_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]]
    for _, row in result.supervised.metrics.iterrows():
        model_rows.append(
            [
                row["Model"],
                format_percent(row["Accuracy"]),
                format_percent(row["Precision"]),
                format_percent(row["Recall"]),
                format_percent(row["F1 Score"]),
                format_percent(row["ROC AUC"]),
            ]
        )
    story.extend([Paragraph("Model Performance", h2), _styled_table(model_rows, col_widths=[1.65 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch, 0.9 * inch])])

    cluster_rows = [["Segment", "Customers", "Monthly Charges", "Tenure", "Churn Risk"]]
    for _, row in result.segmentation.cluster_profile.iterrows():
        cluster_rows.append(
            [
                row["Cluster_Name"],
                f"{int(row['Customer_Count']):,}",
                format_currency(row["Avg_Monthly_Charges"], currency, exchange_rate),
                f"{row['Avg_Tenure']:.1f} months",
                format_percent(row["Churn_Risk"]),
            ]
        )
    story.extend([Paragraph("Customer Segments", h2), _styled_table(cluster_rows, col_widths=[1.85 * inch, 1.0 * inch, 1.35 * inch, 1.0 * inch, 1.0 * inch])])
    story.append(Spacer(1, 12))
    story.append(Paragraph("Built with Streamlit, Python, Machine Learning, and Data Analytics.", body))

    doc.build(story)
    return buffer.getvalue()


def _styled_table(rows: list[list[Any]], col_widths: list[float]) -> Table:
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_excel_report(result: AnalyticsResult, cleaned_df: pd.DataFrame, currency: str, exchange_rate: float) -> bytes:
    metrics_df = pd.DataFrame(
        [
            {"Metric": key, "Value": value}
            for key, value in result.business_metrics.items()
            if key not in {"Business Health Label", "Best Machine Learning Model"}
        ]
    )
    insights_df = pd.DataFrame({"Executive Insight": result.executive_insights})
    predictions = result.analysis_df[
        [
            col
            for col in [
                "customerID",
                "Churn_Probability",
                "Risk_Level",
                "Cluster_Name",
                "MonthlyCharges",
                "TotalCharges",
                "Predicted_Churn",
                "Retention_Offer",
            ]
            if col in result.analysis_df.columns
        ]
    ].copy()
    if currency == "SAR":
        for col in ["MonthlyCharges", "TotalCharges"]:
            if col in predictions.columns:
                predictions[col] = predictions[col] * exchange_rate
    return dataframe_to_excel_bytes(
        {
            "Executive Metrics": metrics_df,
            "Model Comparison": result.supervised.metrics,
            "Predictions": predictions,
            "Clusters": result.segmentation.cluster_profile,
            "Risk Summary": result.risk_summary,
            "Executive Insights": insights_df,
            "Clean Dataset": cleaned_df,
        }
    )


def build_chart_archive(charts: dict[str, Any]) -> tuple[bytes, str]:
    buffer = BytesIO()
    mime = "application/zip"
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, fig in charts.items():
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.lower())
            try:
                archive.writestr(f"{safe}.png", fig.to_image(format="png", scale=2))
            except Exception:
                archive.writestr(f"{safe}.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
    return buffer.getvalue(), mime
