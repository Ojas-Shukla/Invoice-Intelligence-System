import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


@st.cache_resource
def load_models():
    model_dir = Path("models")
    freight_model = None
    flag_model = None
    scaler = None
    if (model_dir / "predict_freight_model.pkl").exists():
        freight_model = joblib.load(model_dir / "predict_freight_model.pkl")
    if (model_dir / "predict_flag_invoice.pkl").exists():
        flag_model = joblib.load(model_dir / "predict_flag_invoice.pkl")
    if (model_dir / "scaler.pkl").exists():
        scaler = joblib.load(model_dir / "scaler.pkl")
    return freight_model, flag_model, scaler


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def prepare_flag_features_from_inputs(values: dict) -> np.ndarray:
    ordered = [
        values.get("invoice_quantity", 0),
        values.get("invoice_dollars", 0.0),
        values.get("freight", values.get("Freight", 0.0)),
        values.get("total_item_quantity", 0),
        values.get("total_item_dollars", 0.0),
    ]
    return np.array([ordered])


def main():
    st.set_page_config(page_title="Invoice Intelligence", page_icon="📊", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; }
        .app-title { font-size: 3rem; font-weight: 800; margin-bottom: 0; }
        .app-subtitle { font-size: 1.05rem; color: #5a6378; margin-top: 0.2rem; }
        .metric-card, .content-card { background: #ffffff; border-radius: 20px; box-shadow: 0 20px 45px rgba(15, 23, 42, 0.05); padding: 24px; }
        .status-badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-size: 0.9rem; margin-right: 8px; font-weight: 600; }
        .status-ok { background: #d1e7dd; color: #0f5132; }
        .status-missing { background: #f8d7da; color: #842029; }
        .streamlit-expanderHeader { font-weight: 700; }
        .stButton>button, .stDownloadButton>button { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='app-title'>Invoice Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>A clean, interactive dashboard to predict freight costs and flag invoice risk with ML.</div>", unsafe_allow_html=True)

    freight_model, flag_model, scaler = load_models()

    with st.sidebar:
        st.markdown("## Navigate")
        mode = st.radio("", ["Freight Prediction", "Invoice Flagging", "Batch Processing"])
        st.markdown("---")
        st.markdown("### Models status")
        st.markdown(
            f"<span class='status-badge {'status-ok' if freight_model else 'status-missing'}'>{'Freight model ready' if freight_model else 'Freight model missing'}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='status-badge {'status-ok' if flag_model and scaler else 'status-missing'}'>{'Flag model ready' if flag_model and scaler else 'Flag model missing'}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("Use the sidebar to switch between prediction modes and upload batch invoice files.")

    if mode == "Freight Prediction":
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Freight Cost Prediction")
        st.write("Predict freight using invoice amount and quantity with a transparent model.")

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("### Input details")
            dollars = st.number_input("Invoice Amount ($)", min_value=0.0, value=1000.0, step=1.0)
            quantity = st.number_input("Invoice Quantity", min_value=0, value=1, step=1)
            if st.button("Predict Freight", key="freight_predict"):
                if freight_model is None:
                    st.error("Model missing — cannot predict.")
                else:
                    n_in = getattr(freight_model, 'n_features_in_', None)
                    if n_in == 2:
                        inp = np.array([[dollars, quantity]])
                    elif n_in == 1:
                        inp = np.array([[dollars]])
                    else:
                        inp = np.array([[dollars, quantity]])
                    pred = freight_model.predict(inp)[0]

                    st.session_state.freight_result = {
                        'dollars': dollars,
                        'quantity': quantity,
                        'predicted_freight': pred,
                    }

        with right:
            st.markdown("### Prediction result")
            if st.session_state.get('freight_result'):
                result = st.session_state.freight_result
                st.metric("Invoice Amount", f"${result['dollars']:,.2f}")
                st.metric("Invoice Quantity", f"{result['quantity']}")
                st.metric("Predicted Freight", f"${result['predicted_freight']:,.2f}")
                fig = px.scatter(
                    x=[result['dollars']],
                    y=[result['predicted_freight']],
                    labels={"x": "Invoice Amount ($)", "y": "Freight ($)"},
                    title="Estimated Freight Cost"
                )
                fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#fafbfc")
                fig.update_traces(marker=dict(size=16, color="#0d6efd"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enter invoice amount and quantity, then click Predict Freight.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif mode == "Invoice Flagging":
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Invoice Risk Assessment")
        st.write("Assess invoice risk with the trained classifier and understand the result at a glance.")

        c1, c2, c3 = st.columns(3)
        with c1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=0, value=1)
            invoice_dollars = st.number_input("Invoice Amount ($)", min_value=0.0, value=1000.0)
        with c2:
            freight = st.number_input("Freight ($)", min_value=0.0, value=100.0)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=0, value=10)
        with c3:
            total_item_dollars = st.number_input("Total Item Amount ($)", min_value=0.0, value=5000.0)

        if st.button("Assess Risk"):
            if flag_model is None or scaler is None:
                st.error("Flag model or scaler missing in models/ — cannot assess risk.")
            else:
                values = {
                    "invoice_quantity": invoice_quantity,
                    "invoice_dollars": invoice_dollars,
                    "freight": freight,
                    "total_item_quantity": total_item_quantity,
                    "total_item_dollars": total_item_dollars,
                }
                features = prepare_flag_features_from_inputs(values)
                features_scaled = scaler.transform(features)
                flag = flag_model.predict(features_scaled)[0]
                proba = flag_model.predict_proba(features_scaled)[0]

                status = "🚨 HIGH RISK — Invoice Flagged" if flag == 1 else "✅ LOW RISK — Invoice Clear"
                if flag == 1:
                    st.error(status)
                else:
                    st.success(status)

                st.metric("Risk Score", f"{max(proba) * 100:.1f}%")

                st.markdown("#### Model inputs")
                df_feat = pd.DataFrame(values, index=[0]).T
                df_feat.columns = ["Value"]
                st.dataframe(df_feat, use_container_width=True)

                fig = go.Figure(data=[go.Bar(x=["Low Risk", "High Risk"], y=proba * 100, marker_color=["#198754", "#dc3545"])])
                fig.update_layout(
                    title="Risk Probability",
                    yaxis_title="Probability (%)",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complete the invoice fields and click Assess Risk to see the result.")
        st.markdown("</div>", unsafe_allow_html=True)

    elif mode == "Batch Processing":
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Batch Invoice Processing")
        st.write("Upload a CSV or Excel file with invoice data to predict freight and flag risky invoices in bulk.")

        uploaded = st.file_uploader("Choose CSV/XLSX", type=["csv", "xlsx"])
        st.markdown("**Required columns:** invoice_quantity, invoice_dollars, freight, total_item_quantity, total_item_dollars")

        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                st.subheader("Preview")
                st.dataframe(df.head(), use_container_width=True)

                proc_df = normalize_columns(df)
                amount_col = 'invoice_dollars' if 'invoice_dollars' in proc_df.columns else ('dollars' if 'dollars' in proc_df.columns else None)
                qty_col = 'invoice_quantity' if 'invoice_quantity' in proc_df.columns else ('quantity' if 'quantity' in proc_df.columns else None)
                flag_cols = ['invoice_quantity', 'invoice_dollars', 'freight', 'total_item_quantity', 'total_item_dollars']
                missing_flag_cols = [c for c in flag_cols if c not in proc_df.columns]

                if amount_col is None or qty_col is None:
                    st.warning("Please include both invoice amount and quantity columns for freight prediction.")
                if missing_flag_cols:
                    st.warning(f"Missing flagging columns: {', '.join(missing_flag_cols)}")

                if st.button("Process File"):
                    if freight_model is not None and amount_col and qty_col:
                        n_in = getattr(freight_model, 'n_features_in_', None)
                        if n_in == 2:
                            freight_X = proc_df[[amount_col, qty_col]].values
                        else:
                            freight_X = proc_df[[amount_col]].values
                        try:
                            proc_df['predicted_freight'] = freight_model.predict(freight_X)
                        except Exception:
                            st.warning("Could not predict freight for batch — check model input shape.")
                            proc_df['predicted_freight'] = np.nan
                    else:
                        proc_df['predicted_freight'] = np.nan

                    if flag_model is not None and scaler is not None and not missing_flag_cols:
                        flag_features = proc_df[flag_cols].values
                        try:
                            flag_scaled = scaler.transform(flag_features)
                            proc_df['flag'] = flag_model.predict(flag_scaled)
                            proc_df['risk_probability'] = flag_model.predict_proba(flag_scaled)[:, 1]
                        except Exception:
                            st.warning("Could not compute invoice flags — check feature order and scaler compatibility.")
                            proc_df['flag'] = np.nan
                            proc_df['risk_probability'] = np.nan
                    else:
                        proc_df['flag'] = np.nan
                        proc_df['risk_probability'] = np.nan

                    st.success("Processing complete")
                    st.markdown("### Results overview")
                    total = len(proc_df)
                    flagged = int(proc_df['flag'].sum() if 'flag' in proc_df else 0)
                    clear = total - flagged
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total invoices", total)
                    c2.metric("Flagged invoices", flagged)
                    c3.metric("Clear invoices", clear)

                    st.subheader("Processed data")
                    st.dataframe(proc_df.head(), use_container_width=True)

                    csv = proc_df.to_csv(index=False)
                    st.download_button("Download Results (CSV)", data=csv, file_name="invoice_results.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error reading file: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
