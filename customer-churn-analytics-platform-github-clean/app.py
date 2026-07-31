from __future__ import annotations

import traceback

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts
from src.analytics import (
    AnalyticsResult,
    SUPPORTED_EXTENSIONS,
    dataframe_to_csv_bytes,
    dataset_summary,
    detect_quality_issues,
    format_currency,
    format_percent,
    predict_manual_customer,
    read_dataset,
    run_full_analytics,
    smart_clean_dataset,
)
from src.reporting import build_chart_archive, build_excel_report, build_executive_pdf
from src.ui import (
    STEP_LABELS,
    alert,
    chart_card,
    format_metric_value,
    inject_css,
    metric_card,
    quality_issue_card,
    render_hero,
    render_progress,
    section,
    summary_table,
)


st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "current_step": 1,
        "max_completed": 0,
        "theme": "Light",
        "currency": "SAR",
        "exchange_rate": 3.75,
        "use_live_rate": False,
        "raw_df": None,
        "cleaned_df": None,
        "file_summary": None,
        "cleaning_summary": None,
        "analytics_result": None,
        "analytics_error": None,
        "dataset_fingerprint": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def dataframe_fingerprint(df: pd.DataFrame) -> int:
    return int(pd.util.hash_pandas_object(df, index=True).sum())


def get_analytics() -> AnalyticsResult | None:
    cleaned = st.session_state.get("cleaned_df")
    if cleaned is None:
        return None
    fingerprint = dataframe_fingerprint(cleaned)
    if st.session_state.analytics_result is not None and st.session_state.dataset_fingerprint == fingerprint:
        return st.session_state.analytics_result
    try:
        with st.spinner("Training churn models, scoring customers, and building customer segments..."):
            st.session_state.analytics_result = run_full_analytics(cleaned)
            st.session_state.analytics_error = None
            st.session_state.dataset_fingerprint = fingerprint
    except Exception as exc:
        st.session_state.analytics_result = None
        st.session_state.analytics_error = f"{exc}\n\n{traceback.format_exc()}"
    return st.session_state.analytics_result


def convert_display_df(df: pd.DataFrame, currency: str, exchange_rate: float) -> pd.DataFrame:
    out = df.copy()
    if currency == "SAR":
        for col in ["MonthlyCharges", "TotalCharges", "Revenue_At_Risk", "Avg_Monthly_Charges", "Avg_Spending"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") * exchange_rate
    return out


def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-logo">
  <h2>Customer Churn Analytics</h2>
  <p>Enterprise BI Command Center</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Workflow Progress")
        for idx, label in enumerate(STEP_LABELS, start=1):
            disabled = idx > max(st.session_state.max_completed + 1, st.session_state.current_step)
            prefix = "✓" if idx <= st.session_state.max_completed else "●" if idx == st.session_state.current_step else "⌁"
            if st.button(f"{prefix} {label}", key=f"nav_{idx}", disabled=disabled, use_container_width=True):
                st.session_state.current_step = idx
                st.rerun()

        st.divider()
        st.caption("Currency")
        st.session_state.currency = st.segmented_control(
            "Currency Toggle",
            options=["SAR", "USD"],
            default=st.session_state.currency,
            label_visibility="collapsed",
            help="Switches all financial values between Saudi Riyal and US Dollar without clearing filters.",
        )
        st.caption("Theme")
        st.session_state.theme = st.segmented_control(
            "Theme Toggle",
            options=["Light", "Dark"],
            default=st.session_state.theme,
            label_visibility="collapsed",
        )

        with st.expander("Settings", expanded=False):
            st.session_state.exchange_rate = st.number_input(
                "Exchange rate: 1 USD = SAR",
                min_value=0.01,
                value=float(st.session_state.exchange_rate),
                step=0.05,
                help="Default is 3.75 SAR per USD.",
            )
            st.session_state.use_live_rate = st.toggle(
                "Use Live Exchange Rate",
                value=st.session_state.use_live_rate,
                help="Optional placeholder. Manual rate is used unless you connect an exchange-rate API.",
            )
            if st.session_state.use_live_rate:
                st.info("Live exchange-rate retrieval is ready to connect; the app is currently using the manual rate.")

        with st.expander("Dataset Information", expanded=True):
            summary = st.session_state.get("file_summary")
            if summary:
                st.write(f"Rows: **{summary['Number of Rows']:,}**")
                st.write(f"Columns: **{summary['Number of Columns']:,}**")
                st.write(f"Missing: **{summary['Missing Values']:,}**")
                st.write(f"Duplicates: **{summary['Duplicate Rows']:,}**")
            else:
                st.caption("No dataset uploaded yet.")

        with st.expander("Quick Statistics", expanded=False):
            result = st.session_state.get("analytics_result")
            if result:
                metrics = result.business_metrics
                st.write(f"Churn Rate: **{format_percent(metrics['Churn Rate'])}**")
                st.write(f"Revenue at Risk: **{format_currency(metrics['Revenue at Risk'], st.session_state.currency, st.session_state.exchange_rate)}**")
                st.write(f"Health: **{metrics['Business Health Score']} / 100**")
            else:
                st.caption("Available after analytics are generated.")

        with st.expander("Export Center", expanded=False):
            st.caption("Reports and data extracts are available in Step 4.")

        with st.expander("About Platform", expanded=False):
            st.caption("Customer Churn Analytics Platform")
            st.caption("Built with Streamlit • Python • Machine Learning • Data Analytics")


def top_bar() -> None:
    left, right = st.columns([0.66, 0.34], vertical_alignment="center")
    with left:
        st.markdown("#### Customer Churn Analytics Platform")
        st.caption("Enterprise customer intelligence, churn prediction, segmentation, and revenue risk analytics")
    with right:
        pill = "Saudi Riyal (SAR)" if st.session_state.currency == "SAR" else "US Dollar (USD)"
        st.markdown(f"**Currency:** {pill} &nbsp;&nbsp; **Theme:** {st.session_state.theme}", unsafe_allow_html=True)


def step_upload() -> None:
    render_hero()
    section("Dataset Upload", "Upload a CSV, XLS, or XLSX customer churn dataset to begin the governed analytics workflow.")
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Drag and drop your dataset here",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=False,
        help="Supported file formats only: CSV (.csv), Excel (.xls), Excel (.xlsx).",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if upload is not None:
        filename = upload.name
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            alert("error", "❌ Invalid File Type", "Only CSV, XLS, and XLSX datasets are supported.")
            return
        try:
            df = read_dataset(upload, filename)
            summary = dataset_summary(df, filename, getattr(upload, "size", 0))
            st.session_state.raw_df = df
            st.session_state.file_summary = summary
            st.session_state.max_completed = max(st.session_state.max_completed, 1)
            st.session_state.cleaned_df = None
            st.session_state.analytics_result = None
            alert("success", "✅ Dataset uploaded successfully.", "The dataset passed file validation and is ready for exploration.")
            summary_table("Dataset Summary", summary)
            if st.button("Continue →", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        except Exception as exc:
            alert("error", "❌ Invalid File Type", f"Only CSV, XLS, and XLSX datasets are supported. Details: {exc}")


def step_eda() -> None:
    df = st.session_state.get("raw_df")
    if df is None:
        alert("warning", "Dataset Required", "Upload a dataset in Step 1 before opening data exploration.")
        return

    section("Exploratory Data Analysis", "Automated profiling, quality diagnostics, and smart cleaning controls.")
    summary = dataset_summary(df, st.session_state.file_summary["File Name"] if st.session_state.file_summary else "Uploaded dataset")
    target_fig = charts.class_distribution(df)

    cols = st.columns(4)
    kpis = [
        ("Total Rows", f"{summary['Number of Rows']:,}", "Number of customer records in the uploaded dataset."),
        ("Total Columns", f"{summary['Number of Columns']:,}", "Number of available dataset fields."),
        ("Missing Values", f"{summary['Missing Values']:,}", "Blank, null, or whitespace-only values detected."),
        ("Duplicate Rows", f"{summary['Duplicate Rows']:,}", "Exact duplicate records detected."),
        ("Numerical Features", f"{summary['Numerical Features']:,}", "Fields stored as numeric values."),
        ("Categorical Features", f"{summary['Categorical Features']:,}", "Fields stored as categories or text."),
        ("Memory Usage", summary["Memory Usage"], "In-memory footprint of the dataset."),
        ("Target Distribution", "Churn", "Distribution of the churn target class."),
    ]
    for i, (label, value, definition) in enumerate(kpis):
        with cols[i % 4]:
            metric_card(label, value, definition)

    section("Data Profiling", "Preview data, inspect structure, and explore statistical relationships.")
    tab_preview, tab_types, tab_stats, tab_corr, tab_missing, tab_dist, tab_box, tab_class = st.tabs(
        ["Dataset Preview", "Data Types", "Descriptive Statistics", "Correlation Heatmap", "Missing Heatmap", "Distribution Plots", "Boxplots", "Class Distribution"]
    )
    with tab_preview:
        st.dataframe(df.head(100), use_container_width=True)
    with tab_types:
        st.dataframe(pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values}), use_container_width=True, hide_index=True)
    with tab_stats:
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    with tab_corr:
        chart_card("Correlation Heatmap", "Shows relationships between numeric features to reveal redundant drivers and business linkages.", charts.correlation_heatmap(df), "correlation_heatmap")
    with tab_missing:
        chart_card("Missing Value Heatmap", "Highlights missing or blank values by feature and sampled records.", charts.missing_heatmap(df), "missing_value_heatmap")
    with tab_dist:
        chart_card("Distribution Plots", "Shows the shape of major numeric features for skew, long tails, and unusual values.", charts.distribution_plot(df), "distribution_plots")
    with tab_box:
        chart_card("Boxplots & Outlier Detection", "Displays extreme values that may influence averages and scaled model features.", charts.boxplot(df), "boxplots")
    with tab_class:
        chart_card("Class Distribution", "Shows whether churn and retention classes are balanced enough for model training.", target_fig, "class_distribution")

    issues, quality_score = detect_quality_issues(df)
    section("Data Quality Analysis", "Each warning includes severity, business impact, and a recommended solution.")
    score_col, action_col = st.columns([0.32, 0.68], vertical_alignment="center")
    with score_col:
        metric_card("Before Cleaning", f"{quality_score} / 100", "Automated quality score before cleaning.", "Data Quality Score", "amber" if quality_score < 80 else "green")
    with action_col:
        if not issues:
            alert("success", "No major data quality issues detected.", "The dataset appears ready for modeling.")
        else:
            issue_cols = st.columns(2)
            for idx, issue in enumerate(issues):
                with issue_cols[idx % 2]:
                    quality_issue_card(issue)

    section("Smart Cleaning", "Applies duplicate removal, missing-value treatment, data type correction, constant-column removal, outlier handling, encoding preparation, and numerical scaling preparation.")
    if st.button("Clean Dataset", use_container_width=True):
        with st.spinner("Cleaning dataset and recalculating data quality score..."):
            cleaned, cleaning_summary = smart_clean_dataset(df)
            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_summary = cleaning_summary
            st.session_state.max_completed = max(st.session_state.max_completed, 2)
            st.session_state.analytics_result = None
        st.rerun()

    if st.session_state.get("cleaned_df") is not None:
        clean_summary = st.session_state.cleaning_summary
        alert("success", "✅ Dataset cleaned successfully.", "The cleaned dataset is ready for executive analytics and machine learning.")
        before, after = st.columns(2)
        with before:
            metric_card("Before Cleaning", f"{clean_summary['Data Quality Before']}%", "Data quality score before smart cleaning.", "Baseline quality", "amber")
        with after:
            metric_card("After Cleaning", f"{clean_summary['Data Quality After']}%", "Data quality score after smart cleaning.", "Improved quality", "green")
        summary_table(
            "Cleaning Summary",
            {
                "Missing values removed": f"{clean_summary['Missing values removed']:,}",
                "Duplicate rows removed": f"{clean_summary['Duplicate rows removed']:,}",
                "Features removed": f"{clean_summary['Features removed']:,}",
                "Corrected data types": ", ".join(clean_summary["Corrected data types"]) or "None",
                "Outlier-handled columns": ", ".join(clean_summary["Outlier-handled columns"]) or "None",
                "Dataset dimensions": f"{clean_summary['Original dimensions']} → {clean_summary['Cleaned dimensions']}",
            },
        )
        st.dataframe(st.session_state.cleaned_df.head(100), use_container_width=True)
        if st.button("Continue →", use_container_width=True, key="eda_continue"):
            st.session_state.current_step = 3
            st.rerun()


def render_business_health(score: int, label: str) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100"},
            title={"text": f"Business Health<br><span style='font-size:0.8em'>{label}</span>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#14b8a6"},
                "steps": [
                    {"range": [0, 60], "color": "rgba(239,68,68,.18)"},
                    {"range": [60, 80], "color": "rgba(245,158,11,.18)"},
                    {"range": [80, 100], "color": "rgba(34,197,94,.18)"},
                ],
                "threshold": {"line": {"color": "#0ea5e9", "width": 4}, "thickness": 0.8, "value": score},
            },
        )
    )
    charts.apply_layout(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def step_executive_dashboard() -> None:
    result = get_analytics()
    if result is None:
        alert("error", "Analytics could not be generated.", st.session_state.analytics_error or "Review dataset columns and try again.")
        return

    currency = st.session_state.currency
    exchange_rate = st.session_state.exchange_rate
    metrics = result.business_metrics

    section("Executive Dashboard", "Premium executive view of churn, retention, customer value, revenue risk, and model readiness.")
    kpi_defs = [
        ("Total Customers", f"{metrics['Total Customers']:,}", "Total customer population in the uploaded dataset.", "Customer base", "blue"),
        ("Churn Rate", format_percent(metrics["Churn Rate"]), "Historical churn rate from the labeled target column.", "Lower is better", "amber" if metrics["Churn Rate"] > 0.22 else "green"),
        ("Retention Rate", format_percent(metrics["Retention Rate"]), "Share of customers retained historically.", "Higher is better", "green"),
        ("Revenue at Risk", format_currency(metrics["Revenue at Risk"], currency, exchange_rate), "Monthly recurring revenue attached to predicted churn customers.", "Predicted exposure", "red"),
        ("Annual Revenue Loss", format_currency(metrics["Annual Revenue Loss"], currency, exchange_rate), "Annualized revenue loss if predicted churn customers leave.", "12 month exposure", "red"),
        ("High-Risk Customers", f"{metrics['High-Risk Customers']:,}", "Customers with High or Critical churn probability.", "Retention queue", "amber"),
        ("Best Machine Learning Model", metrics["Best Machine Learning Model"], "Best display model by F1 score in the comparison table.", "Comparison leader", "teal"),
        ("Best Model Accuracy", format_percent(metrics["Best Model Accuracy"]), "Highest accuracy achieved in the model comparison.", "Accuracy leader", "green"),
        ("Number of Customer Segments", f"{metrics['Number of Customer Segments']:,}", "KMeans customer segments produced by the segmentation pipeline.", "Segmentation depth", "teal"),
        ("Average Customer Lifetime", f"{metrics['Average Customer Lifetime']:.1f} mo", "Average customer tenure in months.", "Lifecycle signal", "blue"),
        ("Average Monthly Charges", format_currency(metrics["Average Monthly Charges"], currency, exchange_rate), "Average monthly customer charge.", "Monthly value", "teal"),
        ("Average Customer Value", format_currency(metrics["Average Customer Value"], currency, exchange_rate), "Average total charges per customer.", "Lifetime value", "teal"),
    ]
    cols = st.columns(4)
    for i, item in enumerate(kpi_defs):
        with cols[i % 4]:
            metric_card(*item)

    section("Business Health Score", "Composite health gauge combining churn, high-risk exposure, and revenue risk.")
    render_business_health(metrics["Business Health Score"], metrics["Business Health Label"])

    section("Revenue Analytics", "Interactive revenue risk and value views. Use chart controls to zoom, pan, and download.")
    dashboard_charts = {
        "revenue_at_risk_by_level": charts.revenue_by_risk(result.analysis_df, currency, exchange_rate),
        "revenue_by_segment": charts.revenue_by_segment(result.analysis_df, currency, exchange_rate),
        "revenue_distribution": charts.revenue_distribution(result.analysis_df, currency, exchange_rate),
        "revenue_trend": charts.revenue_trend(result.analysis_df, currency, exchange_rate),
        "clv_by_segment": charts.clv_by_segment(result.analysis_df, currency, exchange_rate),
        "customer_risk_distribution": charts.risk_distribution(result.analysis_df, donut=True),
    }
    c1, c2 = st.columns(2)
    with c1:
        chart_card("Revenue by Risk Level", "Compares monthly revenue concentration across Low, Medium, High, and Critical risk customers.", dashboard_charts["revenue_at_risk_by_level"], "revenue_by_risk_level")
        chart_card("Revenue Distribution", "Shows how monthly charges are distributed across the customer base.", dashboard_charts["revenue_distribution"], "revenue_distribution")
        chart_card("Customer Lifetime Value", "Compares average lifetime customer value across business-friendly segments.", dashboard_charts["clv_by_segment"], "customer_lifetime_value")
    with c2:
        chart_card("Revenue by Customer Segment", "Shows which customer segments contribute the most monthly recurring revenue.", dashboard_charts["revenue_by_segment"], "revenue_by_segment")
        chart_card("Revenue Trend", "Uses tenure buckets as a customer lifecycle proxy to show revenue momentum.", dashboard_charts["revenue_trend"], "revenue_trend")
        chart_card("Customer Risk Distribution", "Donut view of the customer base across churn risk levels.", dashboard_charts["customer_risk_distribution"], "customer_risk_distribution")

    section("Executive AI Summary", "Clear, non-technical business insights generated from the analytics outputs.")
    insight_cols = st.columns(len(result.executive_insights[:5]))
    for idx, insight in enumerate(result.executive_insights[:5]):
        with insight_cols[idx]:
            metric_card(f"Insight {idx + 1}", "AI", insight, insight[:58] + ("..." if len(insight) > 58 else ""), "teal")

    if st.button("Continue →", use_container_width=True, key="dash_continue"):
        st.session_state.max_completed = max(st.session_state.max_completed, 3)
        st.session_state.current_step = 4
        st.rerun()


def step_ml_analytics() -> None:
    result = get_analytics()
    if result is None:
        alert("error", "Analytics could not be generated.", st.session_state.analytics_error or "Review dataset columns and try again.")
        return
    st.session_state.max_completed = max(st.session_state.max_completed, 4)
    currency = st.session_state.currency
    exchange_rate = st.session_state.exchange_rate

    section("Machine Learning Analytics", "Supervised churn modeling, individual prediction, unsupervised segmentation, and enterprise exports.")
    tab_supervised, tab_prediction, tab_unsupervised, tab_reports = st.tabs(
        ["Supervised Learning", "Customer Prediction", "Unsupervised Learning", "Reports & Downloads"]
    )
    with tab_supervised:
        render_supervised_tab(result)
    with tab_prediction:
        render_prediction_tab(result, currency, exchange_rate)
    with tab_unsupervised:
        render_unsupervised_tab(result, currency, exchange_rate)
    with tab_reports:
        render_reports_tab(result, currency, exchange_rate)


def render_supervised_tab(result: AnalyticsResult) -> None:
    section("Model Evaluation", "Notebook production scoring uses balanced Logistic Regression; the comparison table highlights model tradeoffs.")
    metrics = result.supervised.metrics.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        metrics[col] = metrics[col].map(lambda x: f"{x:.2%}")
    metrics["Training Time"] = metrics["Training Time"].map(lambda x: f"{x:.3f}s")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    best = result.supervised.metrics.iloc[0]
    alert("success", "Best-performing model highlighted", f"{best['Model']} leads the comparison by F1 score. Logistic Regression remains the production predictor from the original notebook workflow.")

    c1, c2 = st.columns(2)
    with c1:
        chart_card("Confusion Matrix", "Shows correct and incorrect churn classifications for the production Logistic Regression model.", charts.confusion_matrix_fig(result.supervised.confusion), "confusion_matrix")
        chart_card("Precision-Recall Curve", "Compares recall and precision tradeoffs for each trained classifier.", charts.precision_recall_fig(result.supervised.curves), "precision_recall_curve")
        chart_card("Feature Importance", "Ranks the strongest production model drivers using absolute Logistic Regression coefficients.", charts.feature_importance_fig(result.supervised.feature_importance), "feature_importance")
        chart_card("Actual vs Predicted", "Compares true churn labels with production model predictions.", charts.actual_vs_predicted(result.supervised.y_test, result.supervised.predictions[result.supervised.production_model_name]), "actual_vs_predicted")
    with c2:
        chart_card("ROC Curve", "Compares model discrimination across thresholds using true and false positive rates.", charts.roc_curve_fig(result.supervised.curves), "roc_curve")
        chart_card("Learning Curve", "Shows how production model F1 changes as training sample size grows.", charts.learning_curve_fig(result.supervised.learning_curve), "learning_curve")
        chart_card("Prediction Distribution", "Shows customer churn probability spread from the production model.", charts.prediction_distribution(result.analysis_df["Churn_Probability"].values), "prediction_distribution")
        chart_card("Class Distribution", "Shows the observed churn class balance used for supervised learning.", charts.class_distribution(result.analysis_df, "Churn"), "supervised_class_distribution")

    section("SHAP Feature Importance", "Available automatically when SHAP is installed and connected to the runtime.")
    alert("warning", "SHAP optional", "The platform is ready for SHAP explanations; if the SHAP package is installed, connect the explainer here without changing the production model.")

    section("AI Business Insights", "Non-technical interpretation of model outputs.")
    insights = [
        "Month-to-month contracts and short tenure should be prioritized in retention campaign design.",
        "High monthly charges increase revenue exposure, so discounting should be targeted rather than broad.",
        "Critical and High risk customers should be routed into a fast-response retention queue.",
        "Feature drivers should be reviewed with sales and service teams before launching offers at scale.",
    ]
    cols = st.columns(4)
    for idx, insight in enumerate(insights):
        with cols[idx]:
            metric_card(f"ML Insight {idx + 1}", "AI", insight, insight[:56] + "...", "teal")


def render_prediction_tab(result: AnalyticsResult, currency: str, exchange_rate: float) -> None:
    section("Customer Prediction", "Manually score a customer using the production Logistic Regression churn model.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with c2:
            tenure = st.number_input("Tenure", min_value=0, max_value=120, value=6, step=1)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with c3:
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input(f"Monthly Charges ({currency})", min_value=0.0, value=350.0 if currency == "SAR" else 90.0, step=5.0)
            total = st.number_input(f"Total Charges ({currency})", min_value=0.0, value=1800.0 if currency == "SAR" else 480.0, step=25.0)
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        with st.expander("Service Profile", expanded=False):
            s1, s2, s3 = st.columns(3)
            with s1:
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            with s2:
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            with s3:
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        submitted = st.form_submit_button("Predict Customer", use_container_width=True)

    if submitted:
        values = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "InternetService": internet,
            "Contract": contract,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "PaperlessBilling": paperless,
            "MultipleLines": multiple_lines,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
        }
        pred = predict_manual_customer(values, result.supervised, currency, exchange_rate)
        cols = st.columns(4)
        with cols[0]:
            metric_card("Prediction", pred["Prediction"], "Binary churn prediction from the production model.", "Manual scoring", "red" if "will churn" in pred["Prediction"] else "green")
        with cols[1]:
            metric_card("Probability", format_percent(pred["Probability"], 0), "Estimated probability of churn.", "Model confidence", "red" if pred["Probability"] >= 0.6 else "green")
        with cols[2]:
            metric_card("Risk", pred["Risk"], "Risk tier based on churn probability thresholds.", "Risk policy", "red" if pred["Risk"] in {"High", "Critical"} else "teal")
        with cols[3]:
            metric_card("Recommendation", "Retention", pred["Recommendation"], pred["Recommendation"], "teal")
        alert("warning" if pred["Risk"] in {"High", "Critical"} else "success", "Top Factors", ", ".join(pred["Top Factors"]))
        alert("success", "Recommended Retention Strategy", pred["Recommendation"])


def render_unsupervised_tab(result: AnalyticsResult, currency: str, exchange_rate: float) -> None:
    section("Clustering Analysis", "Business-friendly customer segments from PCA + KMeans, matching the original notebook segmentation workflow.")
    metrics = result.segmentation.metrics
    cols = st.columns(4)
    for idx, key in enumerate(["Number of Clusters", "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Score"]):
        value = metrics[key]
        display = f"{value:,.3f}" if isinstance(value, float) and pd.notna(value) else f"{value:,}"
        with cols[idx]:
            metric_card(key, display, f"Clustering metric: {key}", "Segment quality", "teal")

    c1, c2 = st.columns(2)
    with c1:
        chart_card("PCA Projection", "Two-dimensional projection used for the KMeans customer segmentation model.", charts.pca_projection(result.segmentation.pca_frame), "pca_projection")
        chart_card("Cluster Size Distribution", "Shows how customers are distributed across named business segments.", charts.cluster_size_distribution(result.analysis_df), "cluster_size_distribution")
        chart_card("Radar Chart", "Compares customer value, tenure, spending, and churn risk across segments.", charts.cluster_radar(result.segmentation.cluster_profile, currency, exchange_rate), "cluster_radar")
    with c2:
        chart_card("Cluster Scatter Plot", "Interactive scatter plot of customer segments in PCA space.", charts.pca_projection(result.segmentation.pca_frame), "cluster_scatter_plot")
        chart_card("Cluster Heatmap", "Normalized profile view for customer count, spending, tenure, and churn risk.", charts.cluster_heatmap(result.segmentation.cluster_profile, currency, exchange_rate), "cluster_heatmap")
        chart_card("Feature Contribution", "Top original encoded features contributing to the PCA segmentation space.", charts.feature_importance_fig(result.segmentation.feature_contribution, "Feature Contribution"), "feature_contribution")

    section("Business-Friendly Cluster Profiles", "Each segment includes customer count, spending, tenure, risk, preferred contract, preferred payment method, and recommended actions.")
    profile = convert_display_df(result.segmentation.cluster_profile, currency, exchange_rate)
    for _, row in profile.iterrows():
        name = row["Cluster_Name"]
        recommendation = cluster_recommendation(name)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(name)
        cols = st.columns(4)
        cards = [
            ("Customer Count", f"{int(row['Customer_Count']):,}", "Customers assigned to this segment."),
            ("Average Spending", format_currency(result.segmentation.cluster_profile.loc[result.segmentation.cluster_profile["Cluster_Name"] == name, "Avg_Spending"].iloc[0], currency, exchange_rate), "Average lifetime spending."),
            ("Average Monthly Charges", format_currency(result.segmentation.cluster_profile.loc[result.segmentation.cluster_profile["Cluster_Name"] == name, "Avg_Monthly_Charges"].iloc[0], currency, exchange_rate), "Average monthly charges."),
            ("Average Tenure", f"{row['Avg_Tenure']:.1f} mo", "Average tenure in months."),
        ]
        for idx, card in enumerate(cards):
            with cols[idx]:
                metric_card(card[0], card[1], card[2], "Cluster KPI", "teal")
        st.write(f"**Churn Risk:** {format_percent(row['Churn_Risk'])}")
        st.write(f"**Preferred Contract:** {row['Preferred_Contract']}")
        st.write(f"**Preferred Payment Method:** {row['Preferred_Payment_Method']}")
        st.write(f"**Business Summary:** {name} represent a distinct behavioral and revenue profile for targeted lifecycle management.")
        st.write(f"**AI Recommendation:** {recommendation}")
        st.markdown("</div>", unsafe_allow_html=True)


def cluster_recommendation(name: str) -> str:
    if "Premium" in name:
        return "Offer loyalty rewards, premium service bundles, and proactive account management."
    if "High-Risk" in name:
        return "Launch targeted retention campaigns with contract migration and service recovery offers."
    if "New" in name:
        return "Improve onboarding experience and guide customers toward value-creating service adoption."
    return "Maintain regular engagement and reinforce long-term loyalty benefits."


def render_reports_tab(result: AnalyticsResult, currency: str, exchange_rate: float) -> None:
    section("Reports & Downloads", "Export cleaned data, predictions, segment assignments, executive reports, Excel workbooks, and dashboard charts.")
    cleaned = st.session_state.get("cleaned_df")
    predictions = result.analysis_df.copy()
    cluster_assignments = result.analysis_df[[col for col in ["customerID", "Cluster", "Cluster_Name"] if col in result.analysis_df.columns]]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Clean Dataset (CSV)", dataframe_to_csv_bytes(cleaned), "clean_dataset.csv", "text/csv", use_container_width=True)
        st.download_button("Predictions (CSV)", dataframe_to_csv_bytes(predictions), "customer_predictions.csv", "text/csv", use_container_width=True)
    with c2:
        st.download_button("Cluster Assignments", dataframe_to_csv_bytes(cluster_assignments), "cluster_assignments.csv", "text/csv", use_container_width=True)
        st.download_button("Executive Report (PDF)", build_executive_pdf(result, currency, exchange_rate), "executive_churn_report.pdf", "application/pdf", use_container_width=True)
    with c3:
        st.download_button("Excel Report", build_excel_report(result, cleaned, currency, exchange_rate), "customer_churn_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        chart_bundle, mime = build_chart_archive(
            {
                "revenue_by_risk_level": charts.revenue_by_risk(result.analysis_df, currency, exchange_rate),
                "revenue_by_segment": charts.revenue_by_segment(result.analysis_df, currency, exchange_rate),
                "customer_risk_distribution": charts.risk_distribution(result.analysis_df, donut=True),
                "pca_projection": charts.pca_projection(result.segmentation.pca_frame),
            }
        )
        st.download_button("Dashboard Charts (PNG)", chart_bundle, "dashboard_charts.zip", mime, use_container_width=True)

    section("Customer Search", "Search by Customer ID and retrieve churn risk, segment, value, and retention recommendation.")
    if "customerID" not in result.analysis_df.columns:
        alert("warning", "Customer ID unavailable", "The uploaded dataset does not include a customerID field.")
        return
    customer_id = st.text_input("Search by Customer ID", placeholder="e.g., 7590-VHVEG")
    if customer_id:
        matches = result.analysis_df[result.analysis_df["customerID"].astype(str).str.contains(customer_id, case=False, na=False)]
        if matches.empty:
            alert("warning", "No customer found", "Try a different customer ID.")
        else:
            row = matches.iloc[0]
            cols = st.columns(3)
            with cols[0]:
                metric_card("Churn Probability", format_percent(row["Churn_Probability"], 0), "Predicted customer churn probability.", "Search result", "red" if row["Churn_Probability"] >= 0.6 else "green")
                metric_card("Risk Level", row["Risk_Level"], "Risk tier based on churn probability.", "Search result", "red" if row["Risk_Level"] in {"High", "Critical"} else "teal")
            with cols[1]:
                metric_card("Cluster", row["Cluster_Name"], "Business-friendly customer segment.", "Search result", "teal")
                metric_card("Monthly Charges", format_currency(row["MonthlyCharges"], currency, exchange_rate), "Customer monthly revenue contribution.", "Search result", "teal")
            with cols[2]:
                metric_card("Total Charges", format_currency(row["TotalCharges"], currency, exchange_rate), "Customer lifetime revenue contribution.", "Search result", "teal")
                metric_card("Revenue Contribution", format_currency(row["MonthlyCharges"] * 12, currency, exchange_rate), "Annualized customer revenue contribution.", "Search result", "blue")
            alert("success", "Recommended Retention Strategy", row["Retention_Offer"])


def footer() -> None:
    st.markdown(
        """
<div class="footer">
  Customer Churn Analytics Platform<br>
  Built with Streamlit • Python • Machine Learning • Data Analytics
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    inject_css(st.session_state.theme)
    sidebar()
    top_bar()
    render_progress(st.session_state.current_step, st.session_state.max_completed)

    if st.session_state.current_step == 1:
        step_upload()
    elif st.session_state.current_step == 2:
        step_eda()
    elif st.session_state.current_step == 3:
        step_executive_dashboard()
    elif st.session_state.current_step == 4:
        step_ml_analytics()
    footer()


if __name__ == "__main__":
    main()
