import streamlit as st
import requests
import pandas as pd
import joblib

st.set_page_config(layout="wide", page_title="Student Dropout Prediction Dashboard")

FEATURE_ORDER = joblib.load('feature_order.pkl')

def load_data():
    return pd.read_csv('cleaned_data_with_flags.csv')

def display_kpi(df):
    total_students = len(df)
    dropout_count = df['dropout'].sum()
    st.metric("Total Students", total_students)
    st.metric("At-risk Students", dropout_count)
    st.metric("Dropout Rate", f"{dropout_count / total_students:.2%}")

    # Histogram of risk scores
    risk_counts = df['risk_score'].value_counts(sort=False).sort_index()
    st.bar_chart(risk_counts)

    # SHAP feature importance
    st.image('shap_summary.png', caption="Feature Importance (SHAP)")

def student_search(df):
    student_id = st.text_input("Search by Student ID")
    if student_id:
        filtered = df[df['student_id'].astype(str).str.contains(student_id)]
        if not filtered.empty:
            st.dataframe(filtered)
        else:
            st.warning("No students found for given ID.")
    else:
        st.dataframe(df[['student_id', 'risk_score', 'poor_attendance_flag', 'academic_stress_flag', 'dropout']])

def student_detail_view(df):
    student_id = st.text_input("Enter Student ID")
    if student_id:
        student_data = df[df['student_id'].astype(str) == student_id]
        if not student_data.empty:
            st.write(student_data)
            record = student_data.iloc[0].to_dict()
            record.pop('dropout', None)
            try:
                response = requests.post('http://localhost:5000/predict', json=record)
                result = response.json()
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(f"Dropout Prediction: {'Yes' if result['prediction'] == 1 else 'No'}")
                    st.write(f"Risk Score: {result['risk_score']:.2f}")
            except requests.exceptions.RequestException:
                st.error("Failed to connect to prediction service.")
        else:
            st.warning("Student ID not found.")
                
def simulation_panel():
    st.header("Simulate Student Data for Risk Prediction")
    inputs = {}
    for feature in FEATURE_ORDER:
        # Assume binary flags and categories, otherwise numeric input
        if 'flag' in feature or feature.endswith(('Yes', 'No')):
            inputs[feature] = st.selectbox(feature, options=[0, 1], index=0)
        else:
            inputs[feature] = st.number_input(feature, value=0)

    if st.button("Simulate Risk Prediction"):
        try:
            response = requests.post('http://localhost:5000/predict', json=inputs)
            result = response.json()
            if 'error' in result:
                st.error(result['error'])
            else:
                st.success(f"Simulated Dropout Risk Score: {result['risk_score']:.2f}")
        except requests.exceptions.RequestException:
            st.error("Failed to connect to prediction service.")

def chatbot_interface():
    st.header("Student Support Chatbot")
    query = st.text_input("Ask a question about student support or system insights")
    if st.button("Send"):
        try:
            response = requests.post('http://localhost:5000/chatbot', json={"query": query})
            res_data = response.json()
            st.write(res_data.get('response', 'No response available'))
        except requests.exceptions.RequestException:
            st.error("Failed to connect to chatbot service.")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["KPI & Analytics", "Students", "Student Details", "Simulation", "Chatbot"])
    
    # Load data once
    df = load_data()

    if page == "KPI & Analytics":
        st.header("Key Performance Indicators and Analytics")
        display_kpi(df)
    elif page == "Students":
        st.header("Search Students Section")
        student_search(df)
    elif page == "Student Details":
        st.header("Student Detail View")
        student_detail_view(df)
    elif page == "Simulation":
        simulation_panel()
    elif page == "Chatbot":
        chatbot_interface()

if __name__ == "__main__":
    main()
