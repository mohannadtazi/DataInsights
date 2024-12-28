import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
from io import StringIO
from fpdf import FPDF


def generate_report(df):
    report = []

    # 1. Initial Data Overview
    report.append("# Dataset Overview")
    report.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    report.append(f"First 5 rows of the dataset:\n{df.head()}")
    report.append("\n")

    # 2. Data Types and Summary
    report.append("## Data Types and Summary Statistics")
    report.append(f"Column Data Types:\n{df.dtypes}")
    report.append(f"Summary Statistics (for numerical columns):\n{df.describe()}")
    report.append("\n")

    # 3. Missing Values Analysis
    missing_values = df.isnull().sum()
    report.append("## Missing Values")
    report.append(missing_values[missing_values > 0].to_string())
    report.append("\n")
    
    # 4. Duplicates
    duplicates = df.duplicated().sum()
    report.append(f"## Duplicates\nThere are {duplicates} duplicate rows in the dataset.")
    report.append("\n")

    # 5. Outliers Detection
    report.append("## Outliers Detection (using Z-Score)")
    z_scores = np.abs((df.select_dtypes(include=[np.number]) - df.mean()) / df.std())
    outliers = (z_scores > 3).sum().sum()
    report.append(f"Total number of outliers detected: {outliers}")
    report.append("\n")

    # 6. Exploratory Data Analysis (EDA)
    report.append("## Exploratory Data Analysis (EDA)")

    # Visualizations: Histograms for numerical columns
    report.append("### Distribution of Numerical Columns:")
    fig = px.histogram(df.select_dtypes(include=[np.number]))
    st.plotly_chart(fig)

    # Correlation matrix
    report.append("### Correlation Matrix:")
    corr_matrix = df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr)

    # 7. Feature Engineering
    report.append("## Feature Engineering")
    categorical_cols = df.select_dtypes(include=['object']).columns
    report.append(f"Categorical columns found: {list(categorical_cols)}")
    
    # Handling categorical data: Encoding
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    report.append("Categorical columns have been label-encoded.")
    
    # Feature scaling
    report.append("### Feature Scaling (Standardization):")
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    report.append("Numerical columns have been standardized.")
    


    # 8. Conclusions and Next Steps
    report.append("## Conclusions and Next Steps")
    report.append("This dataset has been processed and cleaned by DataInsights. Based on the analysis, the next steps would include:")
    report.append("- Further data analysis depending on the business problem.")
    
    # Return the report as a string
    return "\n".join(report)



def generate_pdf(report_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Split the report into lines and add to PDF
    for line in report_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    
    return pdf