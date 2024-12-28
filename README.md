# DataInsights Web App :bar_chart:
DataInsights is an interactive web application built using Streamlit that enables users to upload, clean, preprocess, analyze, and visualize their data. This app allows users to perform Exploratory Data Analysis (EDA) and generate custom reports, making it a useful tool for both beginners and experienced data scientists.

## Features:
- **Data Upload & Management**: Upload CSV or Excel files to work with your data.
- **Data Cleaning & Preprocessing**: Handle missing values, remove duplicates, detect and remove outliers, and convert data types.
- **Exploratory Data Analysis (EDA)**: Generate summary statistics, perform univariate and bivariate analysis, and visualize your data.
- **Data Report Generator**: Automatically generate comprehensive reports on your data.
## Installation:
1. Clone the Repository:
```bash
git clone https://github.com/yourusername/datainsights.git
cd datainsights
```
2. Set Up the Virtual Environment (Optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
3. Install the Required Dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Streamlit App:
```bash
streamlit run app.py
```
## Dependencies:
`streamlit` - For building the web application.
`pandas` - For data manipulation and analysis.
`numpy` - For numerical operations.
`sklearn` - For machine learning models and outlier detection methods.
`plotly` - For interactive visualizations.
`reportlab` - For PDF generation in the report section.
To install the dependencies, you can run:

```bash
pip install streamlit pandas numpy scikit-learn plotly reportlab
```
## Usage:
### 1. Data Upload:

Upload a CSV or Excel file through the sidebar.
The app will automatically load the dataset and show a preview of the data.
### 2. Data Cleaning & Preprocessing:

Analyze and clean your dataset by handling missing values, duplicates, and outliers.
You can also convert data types and apply scaling or encoding.
### 3. Exploratory Data Analysis (EDA):

View summary statistics of your dataset.
Visualize the distribution of individual variables (Univariate Analysis) or the relationship between two variables (Bivariate Analysis).
### 4. Data Report Generator:

Once your data is cleaned and analyzed, generate a report summarizing your data’s key statistics and visualizations.

## File Formats Supported:
CSV (.csv)
Excel (.xlsx)