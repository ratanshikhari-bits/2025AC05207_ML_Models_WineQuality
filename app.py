import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, classification_report

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

st.title("Wine Quality Classification Models")
st.markdown("""
The Machine Learning Model Deployment App! 
Upload the `test_data.csv` file generated during training to evaluate the performance of 5 different classification algorithms on predicting wine quality.
""")

st.header("Upload Test Data")
uploaded_file = st.file_uploader("Upload your test dataset (CSV format)", type=["csv"])

if uploaded_file is not None:

    df_test = pd.read_csv(uploaded_file)
    st.write("Data Preview (Top 5 Rows):", df_test.head())

    X_test = df_test.drop('target', axis=1)
    y_true = df_test['target']

    st.header("2. Model Selection")
    model_dict = {
        "Logistic Regression": "Logistic_Regression",
        "Decision Tree": "Decision_Tree",
        "K-Nearest Neighbors (KNN)": "KNN",
        "Naive Bayes": "Naive_Bayes",
        "Random Forest (Ensemble)": "Random_Forest"
    }
    
    selected_model_display = st.selectbox("Choose an ML Algorithm to evaluate:", list(model_dict.keys()))
    model_file_name = model_dict[selected_model_display]

    if st.button("Run Evaluation"):
        model = joblib.load(f"model/{model_file_name}.pkl")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.header(f"3. Evaluation Metrics: {selected_model_display}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Accuracy", value=round(accuracy_score(y_true, y_pred), 4))
        col2.metric(label="Precision", value=round(precision_score(y_true, y_pred), 4))
        col3.metric(label="Recall", value=round(recall_score(y_true, y_pred), 4))
        
        col4, col5, col6 = st.columns(3)
        col4.metric(label="F1 Score", value=round(f1_score(y_true, y_pred), 4))
        col5.metric(label="MCC", value=round(matthews_corrcoef(y_true, y_pred), 4))
        col6.metric(label="AUC Score", value=round(roc_auc_score(y_true, y_proba), 4))
        
        st.header("4. Model Diagnostics")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad Quality (0)", "Good Quality (1)"])
            disp.plot(cmap='Blues', ax=ax)
            st.pyplot(fig)
            
        with diag_col2:
            st.subheader("Classification Report")
            report = classification_report(y_true, y_pred)
            st.code(report, language="plaintext")
