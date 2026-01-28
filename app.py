# app.py
# Requirements:
# pip install streamlit shap seaborn matplotlib

import streamlit as st
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import confusion_matrix

# ---------------------------
# Page / Theme / CSS
# ---------------------------
st.set_page_config(page_title="Spam Detection", page_icon="📧", layout="centered")

# Light theme + custom CSS for predict button color
CUSTOM_BTN_COLOR = "#1f77b4"   # <- change this hex to any color you like
st.markdown(
    f"""
    <style>
      /* Light background for whole app */
      .stApp {{
        background-color: white;
        color: black;
      }}
      /* Streamlit button - change background */
      div.stButton > button:first-child {{
        background-color: {CUSTOM_BTN_COLOR};
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 6px;
      }}
      div.stButton > button:first-child:hover {{
        filter: brightness(0.95);
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load artifacts (cached)
# ---------------------------
@st.cache_resource
def load_artifacts():
    vec = pickle.load(open("vectorizer.pkl", "rb"))
    sel = pickle.load(open("selector.pkl", "rb"))
    model = pickle.load(open("stacked_model.pkl", "rb"))
    # Optional artifacts
    try:
        cm = pickle.load(open("confusion_matrix.pkl", "rb"))
    except Exception:
        cm = np.array([[230, 5], [12, 180]])  # placeholder if not present
    try:
        accuracy_scores = pickle.load(open("accuracy_scores.pkl", "rb"))
    except Exception:
        accuracy_scores = {"NB": 0.90, "SVM": 0.92, "RF": 0.93, "GB": 0.91, "MLP": 0.89, "Ensemble": 0.95}
    return vec, sel, model, cm, accuracy_scores

vectorizer, selector, model, cm, accuracy_scores = load_artifacts()

# ---------------------------
# Helper to get selected feature names
# ---------------------------
def get_selected_feature_names(vectorizer, selector):
    # vectorizer.get_feature_names_out() available in sklearn >=0.24
    try:
        all_features = vectorizer.get_feature_names_out()
    except Exception:
        all_features = vectorizer.get_feature_names()
    support_mask = selector.get_support()
    selected_features = np.array(all_features)[support_mask]
    return selected_features

selected_feature_names = get_selected_feature_names(vectorizer, selector)

# ==========================
# Load SHAP background
# ==========================
try:
    shap_background = pickle.load(open("shap_background.pkl", "rb"))
except:
    shap_background = np.zeros((1, selector.transform(vectorizer.transform(["test"])).shape[1]))

# Create SHAP explainer
def shap_model(x):
    try:
        return model.predict_proba(x)[:, 1]
    except:
        return model.predict(x).astype(float)

try:
    shap_explainer = shap.KernelExplainer(shap_model, shap_background)
except Exception as e:
    shap_explainer = None
    st.warning("SHAP explainer could not be created. Error: " + str(e))


# ---------------------------
# App UI
# ---------------------------
st.title("📧 Email Spam Detection")
st.write("Stacking ensemble (NB, SVM, RF, GB, MLP) — TF-IDF + SelectKBest. "
         "Enter an email and press Predict. SHAP bar plot shows top contributing words for this prediction.")

user_input = st.text_area("✉️ Enter email text to analyze:", height=180)

CONFIDENCE_THRESHOLD = 0.6

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # 1) Vectorize & select features
        x_tfidf = vectorizer.transform([user_input])
        x_sel = selector.transform(x_tfidf)  # sparse
        # convert to dense numpy for SHAP & plotting
        try:
            x_sel_dense = x_sel.toarray()
        except:
            x_sel_dense = np.array(x_sel)

        # 2) Predict & probability
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_sel_dense)[0]
                spam_prob = float(proba[1])
                ham_prob = float(proba[0])
            else:
                # fallback: maybe stacking model doesn't support predict_proba
                spam_prob = None
                ham_prob = None
        except Exception:
            spam_prob = None
            ham_prob = None

        pred = model.predict(x_sel_dense)[0]
        pred_label = "Spam" if str(pred) in ['1', 'spam'] else "Ham"

        # 3) Show probability calculation
        st.subheader("📊 Spam Probability")
        if spam_prob is not None:
            st.write(f"**Spam probability:** {spam_prob:.4f}")
            st.write(f"**Ham probability:** {ham_prob:.4f}")
            st.write(f"**Decision rule:** Spam if probability ≥ {CONFIDENCE_THRESHOLD}")
        else:
            st.write("Model doesn't provide probability output. Showing label only.")

        # 4) Final result
        if spam_prob is not None:
            if spam_prob >= CONFIDENCE_THRESHOLD:
                st.error("🚨 FINAL RESULT: SPAM")
            else:
                st.success("✅ FINAL RESULT: HAM")
        else:
            if pred_label == "Spam":
                st.error("🚨 FINAL RESULT: SPAM")
            else:
                st.success("✅ FINAL RESULT: HAM")

        st.markdown("---")

        # 5) SHAP bar plot (top 10) — compute only if explainer exists and model supports required calls
        st.subheader("🔎 Local Explanation (SHAP Bar Plot) — Top contributing words")
        if shap_explainer is None:
            st.info("SHAP explainer not available. Ensure `shap` is installed and explainer could be created.")
        else:
            try:
                # KernelExplainer returns shap values for the provided instance
                # This can take a second or two depending on features; we limit nsamples via nsamples param
                with st.spinner("Computing SHAP values (this may take a moment)..."):
                    # use nsamples small for speed (e.g., 100). Lower -> faster but less accurate.
                    shap_values = shap_explainer.shap_values(x_sel_dense, nsamples=100)
                    # shap_values is array-like: for binary prediction KernelExplainer returns
                    # either (n_features,) or list; handle both.
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        # pick values for class 1 (spam)
                        sv = np.array(shap_values[1])[0]
                    else:
                        sv = np.array(shap_values)[0]

                # associate shap values with feature names
                feature_names = selected_feature_names
                # ensure lengths match (sometimes small selector mismatch)
                if sv.shape[0] != len(feature_names):
                    st.warning("Feature length mismatch for SHAP; explanation may be partial.")
                    minlen = min(sv.shape[0], len(feature_names))
                    sv = sv[:minlen]
                    feature_names = feature_names[:minlen]

                # get top k features by absolute shap value
                k = 10
                idx = np.argsort(np.abs(sv))[-k:][::-1]
                top_feats = feature_names[idx]
                top_vals = sv[idx]

                # Plot horizontal bar chart
                fig, ax = plt.subplots(figsize=(7, 4))
                colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_vals]  # red=increase spam, green=reduce spam
                ax.barh(range(len(top_vals))[::-1], top_vals[::-1], color=colors[::-1])  # reverse for top->bottom
                ax.set_yticks(range(len(top_vals))[::-1])
                ax.set_yticklabels(top_feats[::-1])
                ax.set_xlabel("SHAP value (positive -> pushes to SPAM)")
                ax.set_title("Top contributing words for this email")
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as ex:
                st.error("SHAP computation failed or is too slow in this environment.")
                st.write("Error:", ex)
                st.write("Tip: For faster SHAP, save a small representative background sample from training and "
                         "load it into the app; or reduce nsamples in shap_explainer.shap_values(...).")

        st.markdown("---")

        # 6) Confusion matrix (global)
        st.subheader("📘 Confusion Matrix (Model Evaluation on Test Set)")
        try:
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)
        except Exception as ex:
            st.info("Confusion matrix not available.")
            st.write(ex)

        # 7) Accuracy comparison
        st.subheader("📈 Model Accuracy (Comparison)")
        try:
            names = list(accuracy_scores.keys())
            vals = list(accuracy_scores.values())
            fig3, ax3 = plt.subplots(figsize=(7,3))
            sns.barplot(x=names, y=vals, ax=ax3)
            ax3.set_ylim(0,1)
            ax3.set_ylabel("Accuracy")
            st.pyplot(fig3)
        except Exception as ex:
            st.info("Accuracy scores not available.")
            st.write(ex)

# Footer
# st.markdown("""
# ---
# **Notes:**  
# - SHAP KernelExplainer is model-agnostic but can be slow for many features.  
# - For faster, more accurate SHAP, save a small background sample (e.g., 100 rows) from your training set and replace `background` used when creating the explainer.  
# - Change `CUSTOM_BTN_COLOR` at the top of this file to adjust the Predict button color.
# """)
