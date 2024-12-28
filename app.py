import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
import plotly.graph_objects as go
from repport import generate_report, generate_pdf

# Set page title
st.set_page_config(page_title='DataInsights', page_icon=':bar_chart:')

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type.")
        return None

@st.cache_data
def describe_data(df):
    return df.describe()


st.title('DataInsights :bar_chart:')
st.markdown("""
Welcome to **DataInsights**! This interactive web app allows you to upload datasets, perform data cleaning, preprocessing, and exploratory data analysis (EDA). You can also visualize your data and download plots. Enjoy exploring your data! :mag_right:
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([':file_folder: Data Upload & Management', ':wrench: Data Cleaning & Preprocessing', ':mag: Exploratory Data Analysis', ':page_with_curl: Data Report Generator'])



df = pd.DataFrame({
        'Age': np.random.randint(18, 60, size=100),
        'Salary': np.random.randint(30000, 100000, size=100),
        'Department': np.random.choice(['HR', None, 'Marketing'], size=100),
        'Experience': np.random.randint(1, 40, size=100)
    })
with st.sidebar:
    st.title('Upload your dataset :clipboard:')
    uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv','xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
            st.success('CSV file uploaded successfully! :check_mark_button:')
        else:
            df = pd.read_excel(uploaded_file)
            st.success('CSV file uploaded successfully! :check_mark_button:')


    st.download_button(
        label="Download the new dataset as CSV",
        data=df.to_csv().encode('utf-8'),
        file_name="new_dataset.csv",
        mime="text/csv"     
    )

    # Create a multiselect box for selecting columns
    columns = st.multiselect('Select columns', df.columns.tolist())

# Filter the DataFrame based on selected columns
if columns:
    df = df[columns]


if df is not None:
    with tab1:
        st.subheader('Data Preview :file_folder:')
        st.write('Here is a preview of your dataset:')
        
        # Display DataFrame preview with Plotly Table
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='paleturquoise'),
            cells=dict(values=[df[col] for col in df.columns], fill_color='lavender')
        )])
        st.plotly_chart(fig)


        st.info(f'your dataset has {df.shape[0]} rows and {df.shape[1]} columns')

        st.markdown('---')

        st.subheader('Column Data Types :card_index_dividers:')
        st.write('The columns in your dataset are:')
        col_list = df.columns.tolist()
        type_list = df.dtypes.tolist()
        col_type= pd.DataFrame({'column_name':col_list,'data_type':type_list})
        st.dataframe(col_type)
        
        st.markdown('---')

        st.subheader('Summary Statistics :chart_with_upwards_trend:')
        st.write('You can view the summary statistics of your dataset below:')
        st.write(describe_data(df))

        


    with tab2:
        st.subheader('Data Cleaning & Preprocessing :wrench:')
        st.write('You can perform data cleaning and preprocessing here.')
        st.markdown('### 1. Missing Values Analysis :warning:')
        missing_values = df.isnull().sum()
        st.write('The number of missing values in each column are:')
        st.dataframe(missing_values)
        if missing_values.sum() > 0:
            st.write('Detailed Analysis of Columns with Missing Values:')
    
            # Analyze columns with missing values
            for col, count in missing_values.items():
                if count > 0:
                    st.markdown(f"**---   Column: {col}   ---**")
                    st.write(f'- Number of missing values: {count}')
                    st.write(f'- Percentage of missing values: {count / len(df) * 100:.2f}%')

                    if df[col].dtype == 'object':
                        st.write('- Imputation strategy: **mode** for categorical column.')
                        if st.button('Impute missing values'):
                            df[col].fillna(df[col].mode()[0], inplace=True)
                            st.success(f"Missing values in '{col}' imputed with mode strategy.")
                    else:
                        # Calculate standard deviation
                        col_std = df[col].std()
                        st.write(f'- Standard deviation: {col_std:.2f}')
                
                        # Decide imputation strategy
                        threshold = 10  # Define your threshold for high variability
                        if col_std > threshold:
                            st.write('- Using **median** for imputation due to high variability.')
                            if st.button('Impute missing values'):
                                df[col].fillna(df[col].median(), inplace=True)
                                st.success(f"Missing values in '{col}' imputed with median strategy.")
                        else:
                            st.write('- Using **mean** for imputation.')
                            if st.button('Impute missing values'):
                                df[col].fillna(df[col].mean(), inplace=True)
                                st.success(f"Missing values in '{col}' imputed with mean strategy.")

        st.markdown('---')
        st.markdown('### 2. Remove Duplicate Rows :recycle:')
        if df.duplicated().sum() > 0:
            st.write('There are duplicate rows in the dataset.')
            st.write('Number of duplicate rows:', df.duplicated().sum())
            if st.button('Remove duplicates :recycle:'):
                df.drop_duplicates(inplace=True)
                st.success('Duplicate rows removed. :check_mark_button:')
        else:
            st.write("No duplicate rows found. :thumbsup:")

        st.markdown('---')
        st.markdown('### 3. Outlier Detection :boom:')
        st.write('You can detect outliers in your dataset using the following methods:')
        # List of methods
        outlier_methods = ['Z-score method', 'IQR method', 'Isolation Forest method', 'Local Outlier Factor method']
        selected_method = st.selectbox('Select an outlier detection method:', outlier_methods)
        outliers = None
        numerical_cols = col_type[col_type['data_type'] != 'object']['column_name'].tolist()
        if selected_method == 'Z-score method':
            st.write('**Z-Score Method**: Identifies data points that are a certain number of standard deviations away from the mean.')

            threshold = st.slider('Select Z-score threshold', 1.0, 5.0, 3.0)
            z_scores = np.abs((df - df[numerical_cols].mean()) / df.std())
            outliers = (z_scores > threshold).any(axis=1)

            st.write(f'{outliers.sum()} outliers detected using Z-score method.')
            st.dataframe(df[outliers])

        elif selected_method == 'IQR method':
            st.write('**IQR Method**: Identifies outliers based on the Interquartile Range (IQR).')

            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1)

            st.write(f'{outliers.sum()} outliers detected using IQR method.')
            st.dataframe(df[outliers])

        elif selected_method == 'Isolation Forest method':
            st.write('**Isolation Forest Method**: A machine learning-based approach to detect anomalies.')

            contamination = st.slider('Select contamination level', 0.01, 0.5, 0.1)
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(df.select_dtypes(include=[np.number]))
            outliers = outlier_labels == -1

            st.write(f'{outliers.sum()} outliers detected using Isolation Forest method.')
            st.dataframe(df[outliers])

        elif selected_method == 'Local Outlier Factor method':
            st.write('**Local Outlier Factor Method**: Measures the local density deviation of a data point compared to its neighbors.')

            n_neighbors = st.slider('Select number of neighbors', 5, 50, 20)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            outlier_labels = lof.fit_predict(df.select_dtypes(include=[np.number]))
            outliers = outlier_labels == -1

            st.write(f'{outliers.sum()} outliers detected using Local Outlier Factor method.')
            st.dataframe(df[outliers])

        if outliers is not None:
            if st.button('Remove outliers :x:'):
                df = df[~outliers]
                st.success('Outliers have been removed successfully. :check_mark_button:')

        st.markdown('---')
        st.write('# 4. Data Type Conversion')
        st.write('You can convert data types of columns in your dataset.')
        col_type= pd.DataFrame({'column_name':col_list,'data_type':type_list})
        categorical_cols = col_type[col_type['data_type'] == 'object']['column_name'].tolist()
        numerical_cols = col_type[col_type['data_type'] != 'object']['column_name'].tolist()
        st.write('### Categorical columns:')
        categorical_cols_df = pd.DataFrame(categorical_cols, columns=['Categorical Columns'])
        st.dataframe(categorical_cols_df)
        encode_method = st.selectbox('Select encoding method:', ['Label Encoding', 'One-Hot Encoding'])
        st.warning('Please note that `Label Encoding` should be used for ordinal data, while `One-Hot Encoding` should be used for nominal data.')

        if encode_method == 'Label Encoding':
            if st.button('Encode column'):
                df[categorical_cols] = df[categorical_cols].astype('category').cat.codes
                st.success('Column has been encoded successfully!')
                st.write('Encoded DataFrame:')
                st.dataframe(df.head())
        else:
            if st.button('Encode column'):
                df = pd.get_dummies(df, columns=categorical_cols)
                st.success('Column has been encoded successfully!')
                st.write('Encoded DataFrame:')
                st.dataframe(df.head())


        st.markdown('---')
        st.write('# 5. Scaling & Transformation')
        st.write('You can scale and transform numerical columns in your dataset.')
        numerical_cols = col_type[col_type['data_type'] != 'object']['column_name'].tolist()
        st.write('### Numerical columns:')
        numerical_cols_df = pd.DataFrame(numerical_cols, columns=['Numerical Columns'])
        st.dataframe(numerical_cols_df)
        scaling_method = st.selectbox('Select scaling method:', ['Standardization', 'Normalization'])
        st.warning('Please note that `Standardization` should be used for normally distributed data, while `Normalization` should be used for non-normally distributed data.')

        if scaling_method == 'Standardization':
            if st.button('Standardize columns'):
                df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()
                st.success('Columns have been standardized successfully!')
                st.write('Standardized DataFrame:')
                st.dataframe(df.head())
        else:
            if st.button('Normalize columns'):
                df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].min()) / (df[numerical_cols].max() - df[numerical_cols].min())
                st.success('Columns have been normalized successfully!')
                st.write('Normalized DataFrame:')
                st.dataframe(df.head())

                 



    with tab3:
        st.subheader('Exploratory Data Analysis (EDA) :mag:')
        st.write('You can perform exploratory data analysis here.')
        st.markdown('# 1. Summary Statistics :chart_with_upwards_trend:')
        st.write('You can view the summary statistics of your dataset below:')
        st.write(describe_data(df))

        st.markdown('---')
        st.markdown('# 2. Univariate Analysis :bar_chart:')
        st.write('You can analyze individual variables in your dataset.')
        st.write('Select a variable to view its distribution:')
        selected_variable = st.selectbox('Select a variable:', df.columns.tolist())
        st.write('You can customize the appearance of the plot below:')
        plot_type = st.selectbox('Select a plot type:', ['Histogram', 'Boxplot'])
        if plot_type == 'Histogram':
            fig = px.histogram(df, x=selected_variable)
        elif plot_type == 'Boxplot':
            fig = px.box(df, y=selected_variable)

        st.plotly_chart(fig)

        st.markdown('---')
        st.markdown('# 3. Bivariate Analysis :bar_chart:')
        st.write('You can analyze the relationship between two variables in your dataset.')
        st.write('Select two variables to view their relationship:')
        x_variable = st.selectbox('Select X variable:', df.columns.tolist())
        y_variable = st.selectbox('Select Y variable:', df.columns.tolist())
        st.write('You can customize the appearance of the plot below:')
        plot_type = st.selectbox('Select a plot type:', ['Scatter plot', 'Line plot'])
        if plot_type == 'Scatter plot':
            fig = px.scatter(df, x=x_variable, y=y_variable)
        elif plot_type == 'Line plot':
            fig = px.line(df, x=x_variable, y=y_variable)

        st.plotly_chart(fig)

        st.markdown('---')
        st.markdown('# 4. Correlation Analysis :interrobang:')
        st.write('You can analyze the correlation between variables in your dataset.')
        st.write('Select variables to view their correlation:')
        corr_method = st.selectbox('Select correlation method:', ['Pearson', 'Spearman', 'Kendall'])
        st.warning('Pearson correlation coefficient measures the linear relationship between two variables, while Spearman and Kendall.')
        if corr_method == 'Pearson':
            corr = df.corr(method='pearson')
        elif corr_method == 'Spearman':
            corr = df.corr(method='spearman')
        else:
            corr = df.corr(method='kendall')

        fig = px.imshow(corr, title='Correlation Heatmap')
        st.plotly_chart(fig)

        st.markdown('---')
        st.markdown('# 5. Pairwise Analysis')
        st.write('You can analyze the relationship between multiple variables in your dataset.')
        st.write('Select variables to view their relationship:')
        selected_variables = st.multiselect('Select variables:', df.columns.tolist())
        fig = px.scatter_matrix(df[selected_variables])
        st.plotly_chart(fig)

        st.markdown('---')
        st.markdown('# 6. Feature Engineering')
        st.write('You can create new features in your dataset.')
        new_feature_name = st.text_input('Enter new feature name:')
        formula = st.text_input('Enter formula (use column names, e.g., col1 + col2):')

        st.markdown("""
**Instructions for Formula:**
- Use column names from your dataset (case-sensitive).
- Use standard operators (+, -, *, /).
- Example: `col1 + col2 * 0.5`
""")
        #use excel like formula to create new feature
        if st.button('Create new feature') and new_feature_name and formula:
            try:
                # Evaluate the formula using the DataFrame
                df[new_feature_name] = df.eval(formula)
                st.success(f'New feature "{new_feature_name}" has been created successfully!')
                st.write('Updated DataFrame:')
                st.dataframe(df.head())
            except Exception as e:
                st.error(f'Error creating feature: {e}')
        else:
            st.warning('Please provide both a feature name and a formula.')

        

    with tab4:
        st.subheader('Data Report Generator :page_with_curl:')
        st.write('You can generate a detailed data report here.')
        st.write('The report will include the following sections:')
        st.write('1. Data Preview')
        st.write('2. Data Types and Summary Statistics')
        st.write('3. Missing Values Analysis')
        st.write('4. Duplicates')
        st.write('5. Outliers Detection')
        st.write('6. Exploratory Data Analysis (EDA)')
        st.write('7. Feature Engineering')

        if st.button('Generate Data Report'):
            report = generate_report(df)
            st.markdown(report)

            pdf = generate_pdf(report)
            # Create a download link for the PDF
            pdf_output = pdf.output(dest='S').encode('latin1')
                # Allow user to download the report
            st.download_button(
        label="Download Full Report as PDF",
        data=pdf_output,
        file_name="data_report.pdf",
        mime="application/pdf"
    )
        




       