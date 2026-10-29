import hmac
import logging
import os
import traceback
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo
import streamlit as st

from database import (OUTCOMES, clear_history, get_predictions, init_db, save_prediction,
                      save_predictions_bulk, set_actual_outcome)
from ml_utils import (META_PATH, MODEL_PATH, _candidates, csv_safe, dataset_error, dataset_summary,
                      explain_one, find_dataset, history_frame, install_dataset, load_model_bundle_checked,
                      parse_csv_bytes, predict_batch, predict_one, range_warnings, segment_rates)
from trainer import guess_positive_class, guess_target, train_from_dataframe
from ui_compat import wide_kwargs
import auth
import settings

ROOT = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("returnwise")
init_db()


# Cache keys must NOT start with an underscore: Streamlit ignores such arguments when hashing,
# which would stop the caches from refreshing after the files change.
@st.cache_data(show_spinner=False)
def _cached_dataset(file_stamps):
    return find_dataset()


@st.cache_resource(show_spinner=False)
def _cached_model(file_stamps):
    return load_model_bundle_checked()


def _stamps(*paths):
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


W_BUTTON, W_FORM_BTN = wide_kwargs(st.button), wide_kwargs(st.form_submit_button)
W_TABLE, W_CHART = wide_kwargs(st.dataframe), wide_kwargs(st.plotly_chart)


def show_table(data, **kwargs):
    kwargs.pop("use_container_width", None)
    kwargs.setdefault("hide_index", True)
    return st.dataframe(data, **W_TABLE, **kwargs)


def show_chart(fig):
    """theme=None keeps our explicit colours (Streamlit's default theme would recolour the bars)."""
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1e293b"))
    return st.plotly_chart(fig, theme=None, **W_CHART)


st.set_page_config(page_title="ReturnWise | Return Prediction", page_icon="🛍️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #f5f7fb; color: #1e293b; }
/* Keep Streamlit's fixed top bar from covering page titles (it is ~3.75rem tall). */
header[data-testid="stHeader"] { background: rgba(245,247,251,.92); }
[data-testid="stAppDeployButton"], .stAppDeployButton, .stDeployButton { display: none !important; }
.block-container { padding-top: 4.5rem; padding-bottom: 2.5rem; max-width: 1400px; }

/* Main-area text colours (explicit, so an OS dark mode can't turn them white-on-light) */
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] h4, [data-testid="stMain"] h5 { color: #0f172a; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] li, [data-testid="stMain"] label p { color: #1e293b; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: #64748b; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#101d38 0%,#172c52 100%); }
[data-testid="stSidebarHeader"] { padding: .4rem 1rem; height: auto; }
[data-testid="stSidebarUserContent"] { padding-top: .4rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .45rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #f1f5f9; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #94a3b8; }
/* Status boxes live on a pale background, so keep their text dark. */
[data-testid="stSidebar"] [data-testid="stAlert"] p,
[data-testid="stSidebar"] [data-testid="stAlert"] div { color: #0f172a !important; }
/* Navigation buttons: translucent on the dark sidebar with light text. */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.22);
    justify-content: flex-start; text-align: left; min-height: 2.4rem; padding: .3rem .9rem; }
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span { color: #f8fafc !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,.18); border-color: #60a5fa; }
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: #2563eb; border-color: #2563eb; }

/* ---- Cards ---- */
[data-testid="stMetric"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 12px 16px; box-shadow: 0 1px 2px rgba(15,23,42,.05); }
/* Metric labels wrap instead of being cut off with "..." */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 6px; }
[data-testid="stForm"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; }

.brand { font-size: 1.5rem; font-weight: 800; letter-spacing: -.4px; color: #12213f; }
.muted { color: #64748b; margin-bottom: 1rem; }
.hero { padding: 2.1rem 2.2rem; border-radius: 24px; background: linear-gradient(120deg,#102344 0%,#1d4ed8 65%,#60a5fa 100%); }
.hero h1 { color: #fff !important; font-size: clamp(1.7rem, 4vw, 2.6rem); line-height: 1.12; margin: .5rem 0; }
.hero p { color: #e5efff !important; font-size: 1.05rem; max-width: 760px; }
.pill { display: inline-block; padding: 5px 11px; border-radius: 20px; background: #dbeafe; color: #1d4ed8; font-weight: 700; font-size: .78rem; }
div.stButton > button { border-radius: 10px; font-weight: 700; }
div[data-testid="stFormSubmitButton"] button { background: #2563eb; color: #fff; border: 0; }
.small-note { font-size: .86rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ================================================================== authentication
auth.init_users()
try:
    auth.bootstrap_from_env()
except ValueError as exc:
    st.error(f"Could not create the administrator from the environment settings: {exc}")
    st.stop()


def auth_screen():
    """Login (or first-run administrator setup). Hides the sidebar while signed out."""
    st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
                "[data-testid='collapsedControl'] {display: none;}</style>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("<div class='brand'>🛍️ ReturnWise</div><div class='muted'>Online Shopping Return Prediction</div>",
                    unsafe_allow_html=True)
        if auth.user_count() == 0:
            st.info("First-time setup: create the administrator account.")
            with st.form("setup_form"):
                u = st.text_input("Administrator username", key="setup_user")
                p1 = st.text_input(f"Password (at least {auth.MIN_PASSWORD_LENGTH} characters)", type="password", key="setup_pw1")
                p2 = st.text_input("Repeat password", type="password", key="setup_pw2")
                token = st.text_input("Setup token", type="password", key="setup_token") if settings.SETUP_TOKEN else ""
                if st.form_submit_button("Create administrator", **W_FORM_BTN):
                    if settings.SETUP_TOKEN and not hmac.compare_digest(token.encode(), settings.SETUP_TOKEN.encode()):
                        st.error("Incorrect setup token.")
                    elif p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            auth.create_user(u, p1, "admin")
                            st.session_state.flash = "Administrator created. Please sign in."
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
        else:
            flash = st.session_state.pop("flash", None)
            if flash:
                st.success(flash)
            with st.form("login_form"):
                u = st.text_input("Username", key="login_user")
                p = st.text_input("Password", type="password", key="login_pw")
                if st.form_submit_button("Sign in", **W_FORM_BTN):
                    found, message = auth.authenticate(u, p)
                    if found:
                        st.session_state.user = found
                        st.session_state.page = "Welcome"
                        st.rerun()
                    else:
                        st.error(message)
            st.caption("Default login: **admin** / **admin12345** (or set `RETURNWISE_AUTH=off` to bypass)")


auth_on = settings.AUTH_ENABLED and os.environ.get("RETURNWISE_AUTH", "").strip().lower() not in {"off", "0", "false", "no"}
if auth_on:
    user = st.session_state.get("user")
    if user:
        user = auth.get_user(user["id"])            # re-check every run: disabled or demoted users lose access at once
        if user is None or not user["active"]:
            st.session_state.pop("user", None)
            user = None
        else:
            st.session_state.user = user
    if not user:
        auth_screen()
        st.stop()
else:
    user = {"id": 0, "username": "local", "role": "admin", "active": True}
is_admin = bool(user and user.get("role") == "admin")
ADMIN_ONLY = {"Data & Training", "Users"}
setup_page = "Data & Training" if is_admin else "Dashboard"

PAGES = [("Welcome", "⌂"), ("Dashboard", "▦"), ("Data & Training", "⚙"), ("Predict Return", "◎"),
         ("Batch Scoring", "☰"), ("Prediction History", "⏱"), ("Analytics", "▲"), ("Account", "☺"), ("Users", "⚿")]
VISIBLE_PAGES = [(n, i) for n, i in PAGES if is_admin or n not in ADMIN_ONLY]

if "page" not in st.session_state:
    st.session_state.page = "Welcome"


def go(page):
    st.session_state.page = page


with st.sidebar:
    st.markdown("## 🛍️ ReturnWise")
    st.caption("Online Shopping Return Prediction")
    st.divider()
    for item, icon in VISIBLE_PAGES:
        if st.button(f"{icon}  {item}", key="nav_" + item, **W_BUTTON,
                     type="primary" if st.session_state.page == item else "secondary"):
            go(item)
            st.rerun()
    st.divider()
    bundle, model_error = _cached_model(_stamps(MODEL_PATH, META_PATH))
    st.caption("SYSTEM STATUS")
    if bundle:
        st.success("Prediction model loaded")
    elif model_error:
        st.error("Model failed to load")
    else:
        st.warning("Model not trained yet")
    if auth_on:
        st.caption(f"Signed in as **{user['username']}** ({user['role']})")
        if st.button("Sign out", key="logout", **W_BUTTON):
            st.session_state.pop("user", None)
            st.session_state.page = "Welcome"
            st.rerun()
    else:
        st.caption("Login is disabled (local mode)")
    st.caption("ReturnWise • Python + Streamlit")

page = st.session_state.page
if page not in {n for n, _ in VISIBLE_PAGES}:      # server-side access check, not just hidden buttons
    page = st.session_state.page = "Welcome"
df, dataset_path = _cached_dataset(_stamps(*_candidates()))
ds_error = dataset_error() if df is None else None
history = get_predictions()
metrics = (bundle or {}).get("metrics") or {}
POS = (bundle or {}).get("positive_class")


def page_header(title, subtitle):
    st.markdown(f"<div class='brand'>{title}</div><div class='muted'>{subtitle}</div>", unsafe_allow_html=True)


def need_model():
    if bundle:
        return True
    if model_error:
        st.error(model_error)
    else:
        st.warning("No trained model yet. Open **Data & Training** to train one.")
    if is_admin:
        if st.button("Go to Data & Training", type="primary"):
            go("Data & Training")
            st.rerun()
    else:
        st.info("Ask an administrator to upload data and train the model.")
    return False


def dataset_return_rate():
    if df is not None and bundle and bundle["target_column"] in df.columns and POS:
        s = df[bundle["target_column"]].dropna().astype(str).str.strip()
        return float((s == POS).mean()) if len(s) else None
    return None


def show_model_report(b):
    m = b.get("metrics")
    if not m:
        st.info("This model was trained by an older version and has no detailed evaluation. Retrain to see it.")
        return
    st.caption(f"Trained {b.get('trained_at', '?')} on {b.get('n_train', '?')} rows, evaluated on {b.get('n_test', '?')} unseen rows. "
               f"Target: `{b['target_column']}`" + (f" • 'Returned' label: `{b['positive_class']}`" if b.get("positive_class") else ""))
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{m['accuracy']*100:.1f}%", help="Share of test orders classified correctly.")
    cols[1].metric("Baseline accuracy", f"{m['baseline_accuracy']*100:.1f}%",
                   help="Accuracy you would get by always predicting the most common outcome. Judge the model on returns caught / flags that were real, not accuracy alone.")
    if "recall" in m:
        cols[2].metric("Returns caught", f"{m['recall']*100:.1f}%", help="Recall: share of actual returns the model flagged.")
        cols[3].metric("Flags that were real", f"{m['precision']*100:.1f}%", help="Precision: share of flagged orders that really were returned.")
        cols[4].metric("ROC AUC", f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a",
                       help="0.5 = no better than chance, 1.0 = perfect ranking of risky orders.")
        st.caption(f"Orders are flagged as likely returns when estimated probability ≥ {b['decision_threshold']:.2f} "
                   "(tuned on training data to balance precision and recall).")
    for w in b.get("warnings", []):
        if "identifier" in w.lower() or "roc auc" in w.lower():
            continue
        st.warning(w)
    if b.get("version_warning"):
        st.warning(b["version_warning"])
    left, right = st.columns(2)
    with left:
        st.markdown("##### Confusion matrix (test set)")
        labels = b["confusion_labels"]
        fig = px.imshow(b["confusion_matrix"], x=[f"Predicted {l}" for l in labels], y=[f"Actual {l}" for l in labels],
                        text_auto=True, color_continuous_scale="Blues", template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        show_chart(fig)
    with right:
        st.markdown("##### What drives the model (permutation importance)")
        imp = b.get("importances") or []
        if imp:
            idf = pd.DataFrame(imp).head(10).iloc[::-1]
            fig = px.bar(idf, x="importance", y="feature", orientation="h", template="plotly_white")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Drop in score when shuffled")
            show_chart(fig)
        else:
            st.caption("Importance could not be computed.")


# ===================================================================== pages
try:
    if page == "Welcome":
        st.markdown("""
        <div class="hero">
          <span class="pill">MACHINE LEARNING • E-COMMERCE ANALYTICS</span>
          <h1>Smarter decisions.<br>Fewer unexpected returns.</h1>
          <p>Welcome to ReturnWise — a decision-support application that analyses order details and estimates the likelihood of an online shopping return.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        a, b = st.columns([1.2, 1])
        with a:
            st.markdown("### Turn order data into useful signals")
            st.write("Upload your order history, train a model, score single orders or whole files, and compare predictions with what really happened.")
            if st.button("Get Started  →", type="primary", **W_BUTTON):
                go("Predict Return" if bundle else setup_page)
                st.rerun()
            st.markdown("<p class='small-note'>Important: predictions are estimates, not guarantees. Model performance depends on the quality and representativeness of your dataset.</p>", unsafe_allow_html=True)
        with b:
            st.markdown("### Project workflow")
            st.markdown("""
            1. **Data & Training** — upload the order-history CSV and train the model.
            2. **Predict Return** — score one order and see why.
            3. **Batch Scoring** — score a whole file and download the risk list.
            4. **Prediction History** — record what actually happened.
            5. **Dashboard / Analytics** — track accuracy and return patterns.
            """)
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset rows", f"{len(df):,}" if df is not None else "Not found")
        c2.metric("Saved predictions", f"{len(history):,}")
        c3.metric("Model status", "Ready" if bundle else "Needs setup")

    elif page == "Dashboard":
        page_header("Dashboard", "Dataset, model quality and prediction activity at a glance.")
        rate = dataset_return_rate()
        high = sum(1 for h in history if h.get("risk_level") == "High")
        done = [h for h in history if h.get("actual_outcome")]
        c = st.columns(5)
        c[0].metric("Dataset rows", f"{len(df):,}" if df is not None else "—")
        c[1].metric("Historical return rate", f"{rate*100:.1f}%" if rate is not None else "—")
        c[2].metric("Predictions saved", f"{len(history):,}")
        c[3].metric("High-risk flagged", f"{high:,}")
        c[4].metric("Outcomes recorded", f"{len(done):,}")
        if model_error:
            st.error(model_error)
        if ds_error:
            st.error(f"Dataset could not be read — {ds_error}")

        st.markdown("### Guided setup")
        steps = [
            ("Dataset loaded", df is not None, "Data & Training"),
            ("Model trained", bool(bundle), "Data & Training"),
            ("First prediction made", len(history) > 0, "Predict Return"),
            ("Real outcome recorded for a prediction", len(done) > 0, "Prediction History"),
        ]
        for label, ok, target_page in steps:
            a, b = st.columns([4, 1])
            a.markdown(("✅ " if ok else "⬜ ") + label)
            if not ok and (is_admin or target_page not in ADMIN_ONLY) and b.button("Do this", key="step_" + label, **W_BUTTON):
                go(target_page)
                st.rerun()
        if bundle:
            st.markdown("### Model quality")
            show_model_report(bundle)
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### Dataset preview")
            if df is not None:
                st.caption(f"Source: `{dataset_path}`")
                show_table(df.head(8))
            else:
                st.warning("Dataset not found. Upload one in Data & Training.")
        with right:
            st.markdown("### Recent predictions")
            if history:
                show_table(history_frame(history[:8])[["created_at", "prediction", "return_probability", "risk_level"]],
                             )
            else:
                st.info("No predictions yet.")

    elif page == "Data & Training":
        page_header("Data & Training", "Upload order history, choose the 'returned' column, and train the model.")
        st.markdown("### 1. Dataset")
        if df is not None:
            st.success(f"Current dataset: `{dataset_path}` • {df.shape[0]:,} rows × {df.shape[1]} columns")
        elif ds_error:
            st.error(f"Dataset could not be read — {ds_error}")
        else:
            st.info("No dataset yet. Upload a CSV below.")
        up = st.file_uploader("Upload a CSV of historical orders (one row per order, including a returned / not returned column)",
                              type=["csv"], key="dataset_upload")
        if up is not None:
            raw = up.getvalue()
            try:
                new_df = parse_csv_bytes(raw)
            except ValueError as e:
                st.error(str(e))
                new_df = None
            if new_df is not None:
                st.write(f"Read **{len(new_df):,} rows × {len(new_df.columns)} columns**. Preview:")
                show_table(new_df.head(5))
                if st.button("Use this dataset (the current one is backed up)", type="primary"):
                    try:
                        backup = install_dataset(raw)
                        st.success("Dataset saved." + (f" Previous dataset backed up to `{backup.name}`." if backup else ""))
                        st.rerun()
                    except (ValueError, OSError) as e:
                        st.error(f"Could not save the dataset: {e}")

        st.markdown("### 2. Train")
        if df is None:
            st.info("Load a dataset first.")
        else:
            candidates = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 20]
            if not candidates:
                st.error("No column with 2–20 distinct values was found, so there is no usable returned / not-returned column.")
            else:
                guess = guess_target(df, candidates)
                target = st.selectbox("Target column (records whether the order was returned)", candidates,
                                      index=candidates.index(guess))
                classes = sorted(df[target].dropna().astype(str).str.strip().unique().tolist())
                pos_choice = None
                if len(classes) == 2:
                    g = guess_positive_class(classes, df[target].astype(str).str.strip().value_counts().to_dict())
                    pos_choice = st.selectbox("Which value means 'returned'?", classes, index=classes.index(g))
                st.caption(f"Class balance: " + ", ".join(f"{k}: {v:,}" for k, v in
                           df[target].dropna().astype(str).str.strip().value_counts().items()))
                if st.button("Train model", type="primary", **W_BUTTON):
                    with st.spinner("Training and evaluating (this can take a few seconds)…"):
                        try:
                            train_from_dataframe(df, target, positive_class=pos_choice, model_dir=MODEL_PATH.parent)
                            st.session_state.just_trained = True
                        except ValueError as e:
                            st.error(f"Training failed: {e}")
                    if st.session_state.get("just_trained"):
                        st.rerun()
        if st.session_state.pop("just_trained", False):
            st.success("Model trained and saved.")
        if bundle:
            st.markdown("### 3. Current model")
            show_model_report(bundle)

    elif page == "Predict Return":
        page_header("Predict Return", "Enter one order's details to estimate its return risk.")
        if need_model():
            names, ftypes = bundle["feature_names"], bundle["feature_types"]
            defaults = bundle.get("feature_defaults", {})
            int_feats, bool_feats = set(bundle.get("integer_features", [])), set(bundle.get("bool_features", []))
            with st.form("prediction_form"):
                st.markdown("#### Order information")
                cols = st.columns(2)
                values = {}
                for i, name in enumerate(names):
                    label = name.replace("_", " ").title()
                    with cols[i % 2]:
                        d = defaults.get(name)
                        if name in bool_feats:
                            values[name] = st.checkbox(label, value=bool(d), key="feat_" + name)
                        elif ftypes.get(name) == "numeric":
                            if name in int_feats:
                                values[name] = st.number_input(label, value=int(round(d or 0)), step=1, key="feat_" + name)
                            else:
                                values[name] = st.number_input(label, value=float(d or 0.0), format="%.2f", key="feat_" + name)
                        else:
                            options = bundle.get("feature_options", {}).get(name, [])
                            if options and len(options) <= 40:
                                idx = options.index(d) if d in options else 0
                                values[name] = st.selectbox(label, options=options, index=idx, key="feat_" + name)
                            else:
                                values[name] = st.text_input(label, value=d or "", key="feat_" + name)
                submitted = st.form_submit_button("Predict Return Risk", **W_FORM_BTN)
            if submitted:
                try:
                    result = predict_one(bundle, values)
                    notes = range_warnings(bundle, values)
                    effects = explain_one(bundle, values)
                    pid = save_prediction(values, result["label"], result["probability"],
                                          result["risk_probability"], result["risk_level"], "single",
                                          created_by=user["username"])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.info("Check that your inputs match the training dataset, or retrain the model.")
                else:
                    st.divider()
                    left, right = st.columns(2)
                    with left:
                        st.markdown("### Prediction result")
                        if result["risk_probability"] is not None:
                            thr = result["threshold"]
                            fig = pgo.Figure(pgo.Indicator(
                                mode="gauge+number", value=result["risk_probability"] * 100,
                                number={"suffix": "%", "valueformat": ".1f"}, title={"text": "Estimated return probability"},
                                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1e3a8a"},
                                       "steps": [{"range": [0, thr * 60], "color": "#bbf7d0"},
                                                 {"range": [thr * 60, thr * 100], "color": "#fde68a"},
                                                 {"range": [thr * 100, 100], "color": "#fecaca"}]}))
                            fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
                            show_chart(fig)
                            level = result["risk_level"]
                            banner = {"High": st.error, "Medium": st.warning, "Low": st.success}.get(level, st.info)
                            banner(f"**{level or 'Estimated'} risk** — predicted outcome: **{result['label']}**")
                            advice = {"High": "Consider extra checks before dispatch: confirm size/fit guidance, verify the address, or flag for follow-up.",
                                      "Medium": "Worth monitoring; clear product information may reduce the chance of a return.",
                                      "Low": "No special handling suggested."}.get(level, "No special handling suggested.")
                            st.caption(f"Suggested handling (rule of thumb, not a model output): {advice}")
                        else:
                            st.metric("Predicted outcome", result["label"])
                            st.metric("Model confidence", f'{result["probability"]*100:.1f}%')
                        st.caption("Estimates are not guarantees. Scores are only as good as the training data.")
                    with right:
                        st.markdown("### Why this score?")
                        if effects:
                            edf = pd.DataFrame(effects)
                            edf["label"] = edf["feature"] + " = " + edf["value"].astype(str)
                            edf["direction"] = edf["effect"].apply(lambda v: "raises risk" if v > 0 else "lowers risk")
                            fig = px.bar(edf.iloc[::-1], x="effect", y="label", orientation="h", color="direction",
                                         color_discrete_map={"raises risk": "#dc2626", "lowers risk": "#16a34a"},
                                         template="plotly_white")
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="",
                                              xaxis_title="Change in return probability vs a typical order", yaxis_title="")
                            show_chart(fig)
                            st.caption("What-if view: each bar shows how the probability would change if that value were replaced by the typical (median / most common) value.")
                        else:
                            st.info("All values match a typical order, or an explanation isn't available for this model.")
                    for n in notes:
                        st.warning(n + " Predictions here are less reliable.")
                    st.success(f"Saved to history (#{pid}). Record the real outcome later in Prediction History.")

    elif page == "Batch Scoring":
        page_header("Batch Scoring", "Score a whole file of orders and download the ranked risk list.")
        if need_model():
            st.write("Upload a CSV containing these columns (extra columns are kept):")
            st.markdown(" ".join(f"`{c}`" for c in bundle["feature_names"]))
            up = st.file_uploader("Orders to score", type=["csv"], key="batch_upload")
            if up is not None:
                try:
                    new_df = parse_csv_bytes(up.getvalue())
                    scored, notes = predict_batch(bundle, new_df)
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
                else:
                    for n in notes:
                        st.warning(n)
                    if "return_probability" in scored:
                        scored = scored.sort_values("return_probability", ascending=False)
                        c = st.columns(4)
                        c[0].metric("Orders scored", f"{len(scored):,}")
                        c[1].metric("High risk", f"{(scored['risk_level'] == 'High').sum():,}")
                        c[2].metric("Medium risk", f"{(scored['risk_level'] == 'Medium').sum():,}")
                        c[3].metric("Average risk", f"{scored['return_probability'].mean()*100:.1f}%")
                        fig = px.histogram(scored, x="return_probability", nbins=30, color="risk_level", template="plotly_white",
                                           color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"})
                        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Estimated return probability")
                        show_chart(fig)
                    show_table(scored.head(200))
                    if len(scored) > 200:
                        st.caption(f"Showing the first 200 of {len(scored):,} rows; the download contains all rows.")
                    st.download_button("Download scored CSV", csv_safe(scored).to_csv(index=False).encode("utf-8"),
                                       "scored_orders.csv", "text/csv")
                    tcol = bundle["target_column"]
                    if POS and tcol in new_df.columns:
                        actual = new_df.loc[scored.index, tcol].astype(str).str.strip() == POS
                        predicted = scored["predicted_outcome"] == POS
                        st.markdown("#### This file contains actual outcomes — how did the model do?")
                        tp = int((actual & predicted).sum())
                        st.write(f"Caught **{tp:,}** of **{int(actual.sum()):,}** actual returns; "
                                 f"**{int((predicted & ~actual).sum()):,}** flags were false alarms.")
                        st.caption("If this file overlaps the training data the result is optimistic.")
                    if "return_probability" in scored and st.button(f"Save first {min(len(scored), 1000):,} scored rows to history"):
                        feats = bundle["feature_names"]
                        rows = [{"inputs": {f: (None if pd.isna(r[f]) else r[f]) for f in feats},
                                 "prediction": r["predicted_outcome"], "risk_probability": float(r["return_probability"]),
                                 "risk_level": r["risk_level"]} for _, r in scored.head(1000).iterrows()]
                        st.success(f"Saved {save_predictions_bulk(rows, created_by=user['username'])} rows to history.")

    elif page == "Prediction History":
        page_header("Prediction History", "Saved predictions — record what actually happened to measure real accuracy.")
        history = get_predictions()
        if not history:
            st.info("No predictions saved yet. Run a prediction to build your local history.")
        else:
            hdf = history_frame(history)
            levels = [l for l in ["High", "Medium", "Low"] if l in set(hdf["risk_level"].dropna())]
            pick = st.multiselect("Filter by risk level", levels, default=levels) if levels else []
            view = hdf[hdf["risk_level"].isin(pick) | hdf["risk_level"].isna()] if levels else hdf
            show_table(view)
            st.download_button("Download history as CSV", csv_safe(hdf).to_csv(index=False).encode("utf-8"),
                               "prediction_history.csv", "text/csv")

            st.markdown("### Record the real outcome")
            with st.form("outcome_form"):
                a, b = st.columns(2)
                pid = a.selectbox("Prediction #", [h["id"] for h in history])
                outcome = b.selectbox("What actually happened?", list(OUTCOMES))
                if st.form_submit_button("Save outcome", **W_FORM_BTN):
                    if set_actual_outcome(pid, outcome):
                        st.success(f"Saved outcome for #{pid}.")
                        st.rerun()
                    else:
                        st.error("That prediction no longer exists.")
            done = [h for h in history if h.get("actual_outcome") and h.get("risk_level")]
            if done and POS:
                correct = sum((h["prediction"] == POS) == (h["actual_outcome"] == "Returned") for h in done)
                c = st.columns(3)
                c[0].metric("With real outcome", len(done))
                c[1].metric("Correct", correct)
                c[2].metric("Real-world accuracy", f"{correct / len(done) * 100:.0f}%")
                if len(done) < 30:
                    st.caption("Fewer than 30 outcomes recorded — treat this figure as a rough indication only.")
            if is_admin:
                st.divider()
                confirm = st.checkbox("I understand this permanently deletes all saved predictions")
                if st.button("Clear prediction history", disabled=not confirm):
                    clear_history()
                    st.rerun()

    elif page == "Analytics":
        page_header("Analytics", "Explore the dataset and see which segments return most.")
        if df is None:
            if ds_error:
                st.error(f"Dataset could not be read — {ds_error}")
            else:
                st.warning("Dataset not found. Upload one in Data & Training.")
        else:
            tcol = bundle["target_column"] if bundle else None
            if bundle and tcol in df.columns and POS:
                st.markdown("### Return rate by segment")
                options = [c for c in df.columns if c != tcol and df[c].nunique(dropna=True) > 1
                           and (pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= 30)]
                if not options:
                    st.info("No suitable columns to break down by.")
                else:
                    col = st.selectbox("Break down by", options)
                    seg = segment_rates(df, col, tcol, POS)
                    overall = (df[tcol].astype(str).str.strip() == POS).mean() * 100
                    fig = px.bar(seg, x="segment", y="return_rate_pct", hover_data=["count"], template="plotly_white",
                                 title=f"Return rate by {col}", text=seg["return_rate_pct"].round(1).astype(str) + "%")
                    fig.update_traces(marker_color="#2563eb", textposition="outside", cliponaxis=False)
                    fig.add_hline(y=overall, line_dash="dash", line_color="#dc2626",
                                  annotation_text=f"Overall {overall:.1f}%", annotation_position="top left")
                    fig.update_layout(yaxis_title="Return rate (%)", xaxis_title=col, margin=dict(t=60),
                                      xaxis={"type": "category"})
                    show_chart(fig)
                st.caption("Small segments (low count) can swing a lot by chance — check the hover counts.")
            else:
                st.info("Train a model first to unlock return-rate breakdowns (it records which column is the target).")
            st.markdown("### Dataset overview")
            show_table(df.describe(include="all").transpose().infer_objects(copy=False).fillna("").astype(str))
        if history:
            st.markdown("### Saved prediction outcomes")
            hdf = history_frame(history)
            counts = hdf["prediction"].value_counts().rename_axis("Prediction").reset_index(name="Count")
            show_chart(px.pie(counts, names="Prediction", values="Count", hole=.45, template="plotly_white"))

    elif page == "Account":
        page_header("My account", "Your sign-in details and password.")
        st.write(f"Signed in as **{user['username']}** — role: **{user['role']}**")
        if not auth_on:
            st.info("Login is disabled in this deployment, so there is no password to change.")
        else:
            with st.form("password_form", clear_on_submit=True):
                old = st.text_input("Current password", type="password", key="pw_old")
                new1 = st.text_input(f"New password (at least {auth.MIN_PASSWORD_LENGTH} characters)", type="password", key="pw_new1")
                new2 = st.text_input("Repeat new password", type="password", key="pw_new2")
                if st.form_submit_button("Change password", **W_FORM_BTN):
                    if new1 != new2:
                        st.error("The new passwords do not match.")
                    else:
                        try:
                            auth.change_password(user["id"], old, new1)
                            st.success("Password changed.")
                        except ValueError as exc:
                            st.error(str(exc))

    elif page == "Users":
        page_header("Users", "Admins can train models and manage users; analysts can predict and review results.")
        flash = st.session_state.pop("flash", None)
        if flash:
            st.success(flash)
        users = auth.list_users()
        show_table(pd.DataFrame(users)[["username", "role", "active", "last_login", "created_at"]])
        st.markdown("### Add a user")
        with st.form("add_user_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            new_name = c1.text_input("Username", key="new_user_name")
            new_role = c2.selectbox("Role", list(auth.ROLES), index=1, key="new_user_role")
            new_pw = st.text_input(f"Temporary password (at least {auth.MIN_PASSWORD_LENGTH} characters)", type="password", key="new_user_pw")
            if st.form_submit_button("Create user", **W_FORM_BTN):
                try:
                    auth.create_user(new_name, new_pw, new_role)
                    st.session_state.flash = f"User '{auth.normalize_username(new_name)}' created."
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        st.markdown("### Manage a user")
        with st.form("manage_user_form", clear_on_submit=True):
            target_name = st.selectbox("User", [u["username"] for u in users], key="manage_user")
            action = st.selectbox("Action", ["Reset password", "Disable account", "Enable account", "Make admin", "Make analyst"],
                                  key="manage_action")
            manage_pw = st.text_input("New password (only for 'Reset password')", type="password", key="manage_pw")
            if st.form_submit_button("Apply", **W_FORM_BTN):
                target = next((u for u in users if u["username"] == target_name), None)
                try:
                    if target is None:
                        raise ValueError("User not found.")
                    if target["id"] == user["id"] and action in ("Disable account", "Make analyst"):
                        raise ValueError("You cannot disable or demote your own account.")
                    if action == "Reset password":
                        auth.reset_password(target["id"], manage_pw)
                    elif action == "Disable account":
                        auth.set_active(target["id"], False)
                    elif action == "Enable account":
                        auth.set_active(target["id"], True)
                    else:
                        auth.set_role(target["id"], "admin" if action == "Make admin" else "analyst")
                    st.session_state.flash = f"Done: {action.lower()} for '{target_name}'."
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
except Exception:  # noqa: BLE001  (st.rerun()/st.stop() are BaseExceptions and pass straight through)
    error_id = uuid.uuid4().hex[:8]
    logger.exception("Unhandled error on page %r (error id %s)", page, error_id)
    st.error(f"Something went wrong while showing this page. Reference: {error_id}")
    if is_admin:
        with st.expander("Technical details (administrators only)"):
            st.code(traceback.format_exc())
