import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots




# --- Web Page Preparation ---
st.set_page_config(page_title="Bank Churn Prediction App")

# Load the model
try:
    classifier = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'random_forest_model.pkl' is present in the directory.")
    st.stop()  # Stop execution if the model cannot be loaded

# Get feature importance from the model
if hasattr(classifier, 'feature_importances_'):
    feature_importances = classifier.feature_importances_
else:
    st.error("The model does not have feature importances. Ensure it is a model that supports feature importance.")
    st.stop()



# --- Setting separate columns for App and Documentation ---

tab1, tab2 = st.tabs(["App", "Documentation"])

with tab1:
    
    # # --- App Input ---
    st.markdown("""
                <h2 style="color: #0CABA8;">Bank Churn Prediction App</h2>
                """,unsafe_allow_html=True)
    st.write("""
    🏦 Welcome to our Bank Churn Prediction Tool! Use this tool to estimate the likelihood of a customer leaving the bank based on factors like account age, 
    transaction behavior, credit utilization, and overall engagement with the bank's services.
    """)
    

    # Input demographic variables
    st.sidebar.markdown(
        """
        <h3 style="color: #0CABA8;">Customer Info</h3>
        """,
        unsafe_allow_html=True
    )

    ip_customer_age = st.sidebar.number_input('Customer Age', min_value=18, max_value=100, value=30)
    ip_gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    ip_dependent_count = st.sidebar.number_input('Dependent Count', min_value=0, max_value=10, value=1)
    ip_education_level = st.sidebar.selectbox('Education Level', ['High School', 'Graduate', 'Post-Graduate', 'Uneducated', 'Unknown'])
    ip_marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    ip_income_category = st.sidebar.selectbox('Income Category', ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', 'More than $120K'])

    # Add a horizontal line using Markdown
    st.sidebar.markdown("---")

    # Input product variables
    st.sidebar.markdown(
        """
        <h3 style="color: #0CABA8;">Card Info</h3>
        """,
        unsafe_allow_html=True
    )
    ip_card_category = st.sidebar.selectbox('Card Category', ['Blue', 'Silver', 'Gold', 'Platinum'])
    ip_months_on_book = st.sidebar.number_input('Months on Book', min_value=0, max_value=60, value=24)
    ip_total_relationship_count = st.sidebar.number_input('Total Relationship Count', min_value=0, max_value=10, value=3)
    ip_months_inactive_12_mon = st.sidebar.number_input('Months Inactive in Last 12 Months', min_value=0, max_value=12, value=0)
    ip_contacts_count_12_mon = st.sidebar.number_input('Contacts Count in Last 12 Months', min_value=0, max_value=12, value=2)

    # Add a horizontal line using Markdown
    st.sidebar.markdown("---")

    # Input credit variables
    st.sidebar.markdown(
        """
        <h3 style="color: #0CABA8;">Credit Info</h3>
        """,
        unsafe_allow_html=True
    )
    ip_credit_limit = st.sidebar.number_input('Credit Limit', min_value=2000, max_value=1000000, value=10000)
    ip_total_revolving_bal = st.sidebar.number_input('Total Revolving Balance', min_value=0, max_value=100000, value=1000)
    ip_avg_open_to_buy = st.sidebar.number_input('Average Open to Buy', min_value=0, max_value=1000000, value=5000)
    ip_avg_utilization_ratio = st.sidebar.number_input('Average Utilization Ratio', min_value=0.0, max_value=1.0, value=0.3)

    # Add a horizontal line using Markdown
    st.sidebar.markdown("---")

    # Input transaction variables
    st.sidebar.markdown(
        """
        <h3 style="color: #0CABA8;">Transaction Info</h3>
        """,
        unsafe_allow_html=True
    )
    ip_total_amt_chng_q4_q1 = st.sidebar.number_input('Total Amount Change (Q4-Q1)', min_value=0.0, max_value=3.0, value=1.0)
    ip_total_trans_amt = st.sidebar.number_input('Total Transaction Amount', min_value=0, max_value=1000000, value=5000)
    ip_total_trans_ct = st.sidebar.number_input('Total Transaction Count', min_value=10, max_value=150, value=50)
    ip_total_ct_chng_q4_q1 = st.sidebar.number_input('Total Transaction Count Change (Q4-Q1)', min_value=0.0, max_value=5.0, value=1.0)

    # Prepare input for the model
    user_input_preprocess = [
        ip_customer_age,
        ip_gender,
        ip_dependent_count,
        ip_education_level,
        ip_marital_status,
        ip_income_category,
        ip_card_category,
        ip_months_on_book,
        ip_total_relationship_count,
        ip_months_inactive_12_mon,
        ip_contacts_count_12_mon,
        ip_credit_limit,
        ip_total_revolving_bal,
        ip_avg_open_to_buy,
        ip_total_amt_chng_q4_q1,
        ip_total_trans_amt,
        ip_total_trans_ct,
        ip_total_ct_chng_q4_q1,
        ip_avg_utilization_ratio
    ]


    # Prepare input for the model
    user_input_postprocess = [
        ip_customer_age,
        1 if ip_gender == "Male" else 0,
        ip_dependent_count,
        1 if ip_education_level == "Uneducated" else 
        2 if ip_education_level == "High School" else
        3 if ip_education_level == "Graduate" else
        4 if ip_education_level == "College" else
        5 if ip_education_level == "Post-Graduate" else
        6 if ip_education_level == "Doctorate" else 0,
        1 if ip_marital_status == "Married" else 
        2 if ip_marital_status == "Unknown" else 0,
        1 if ip_income_category == "$40K - $60K" else
        2 if ip_income_category == "$60K - $80K" else
        3 if ip_income_category == "$80K - $120K" else
        4 if ip_income_category == "$120K +" else 0,
        1 if ip_card_category == "Silver" else
        2 if ip_card_category == "Gold" else
        3 if ip_card_category == "Platinum" else 0,
        ip_months_on_book,
        ip_total_relationship_count,
        ip_months_inactive_12_mon,
        ip_contacts_count_12_mon,
        ip_credit_limit,
        ip_total_revolving_bal,
        ip_avg_open_to_buy,
        ip_total_amt_chng_q4_q1,
        ip_total_trans_amt,
        ip_total_trans_ct,
        ip_total_ct_chng_q4_q1,
        ip_avg_utilization_ratio
    ]

    # Convert input to DataFrame for model prediction
    input_df_preprocess = pd.DataFrame([user_input_preprocess], columns=[
        'customer_age', 'gender', 'dependent_count', 'education_level',
        'marital_status', 'income_category', 'card_category', 'months_on_book',
        'total_relationship_count', 'months_inactive_12_mon',
        'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
        'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
        'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio'
    ])

    input_df_postprocess = pd.DataFrame([user_input_postprocess], columns=[
        'customer_age', 'gender', 'dependent_count', 'education_level',
        'marital_status', 'income_category', 'card_category', 'months_on_book',
        'total_relationship_count', 'months_inactive_12_mon',
        'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
        'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
        'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio'
    ])

    st.write("Your Input:")
    st.dataframe(input_df_preprocess)

    if st.button('Predict'):
        try:
            # Make predictions
            prediction = classifier.predict(input_df_postprocess)
            prediction_proba = classifier.predict_proba(input_df_postprocess)
            
            # Display prediction results
            if prediction[0] == 1:
                st.error("Your customer is likely to churn.")
            else:
                st.success("Yay! Your customer will stay.")
            
            # Display prediction probabilities
            st.write(f"Prediction Probability [Stay, Churn]: {prediction_proba[0]}")
            st.write("*Note*: The Machine Learning Model is heavily influenced by Feature Importance which will be discussed later.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Define feature names
    feature_names = [
        'customer_age', 'gender', 'dependent_count', 'education_level',
        'marital_status', 'income_category', 'card_category', 'months_on_book',
        'total_relationship_count', 'months_inactive_12_mon',
        'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
        'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
        'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio'
    ]


    


with tab2:

    # Define file path
    csv_file_path = r"BankChurners.csv"

    # Check if the file exists
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"The file at {csv_file_path} does not exist.")

    # Load dataset
    df = pd.read_csv(csv_file_path)

    # Get feature importance from the model
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
    else:
        st.error("The model does not have feature importances. Ensure it is a model that supports feature importance.")
        st.stop()

    ### Defining variables

    # Define feature names
    feature_names = [
        'customer_age', 'gender', 'dependent_count', 'education_level',
        'marital_status', 'income_category', 'card_category', 'months_on_book',
        'total_relationship_count', 'months_inactive_12_mon',
        'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
        'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt',
        'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio'
    ]

    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)



    # Set global Matplotlib configuration for transparent background and white font
    plt.rcParams['figure.facecolor'] = 'none'  # Figure background to transparent
    plt.rcParams['axes.facecolor'] = 'none'    # Axes background to transparent
    plt.rcParams['savefig.facecolor'] = 'none' # Background for saved figures
    plt.rcParams['text.color'] = 'white'       # Default text color
    plt.rcParams['axes.labelcolor'] = 'white'  # Axis label color
    plt.rcParams['xtick.color'] = 'white'      # X-axis tick color
    plt.rcParams['ytick.color'] = 'white'      # Y-axis tick color
    plt.rcParams['axes.edgecolor'] = 'white'   # Axes edge color
    plt.rcParams['grid.color'] = 'white'       # Grid line color, if used

    # Renaming all headers with lowercase
    df.columns = df.columns.str.lower()


    # Drop unused columns
    drop_columns = df.drop(columns=['clientnum', 
                                    'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1', 
                                    'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2'],
                                    inplace=True)


    # # --- Table of Contents ---
    # with st.expander("See more"):
    # Set up the Table of Contents
    st.markdown("""
    <style>
        .toc {
            font-family: Arial, sans-serif;
            font-size: 18px;
            line-height: 1.6;
            padding: 10px;
        }
        .toc a {
            text-decoration: none;
            color: #FF6347; /* Tomato color for links */
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .toc h1 {
            font-size: 24px;
            margin-bottom: 10px;
        }
    </style>

    <div class="toc">
        <h1>Table of Contents</h1>
        <ul>
            <li><a href="#introduction">Problem Statement</a></li>
            <li><a href="#data-overview">Data Overview</a></li>
            <li><a href="#eda">EDA</a></li>
            <li><a href="#modeling">Modeling</a></li>
            <li><a href="#conc">Conclusion</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)









    # --- Problem Statement ---

    st.markdown("""
                <a name="introduction"></a>
                
                <h2 style="color: #0CABA8;">Problem Statement</h2>

                **Objective**: The goal is to predict customer attrition for a bank, which refers to whether a customer will close their account with the bank. 
                By analyzing various demographic, product, credit, and transaction attributes of the customers, the objective is to develop a classification model 
                that can accurately predict whether a customer will attrite (close their account) or not. The model will classify customers into these two 
                categories based on the given attributes, helping the bank proactively identify at-risk customers and potentially take steps to retain them.
                
                """, unsafe_allow_html=True)

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)









    # --- Data Overview ---

    st.markdown("""
                <a name="data-overview"></a>

                <h2 style="color: #0CABA8;">Data Overview</h2>

                """, unsafe_allow_html=True)


    st.write("""
    Here we provide an overview of the data. The dataset includes features and analyzes how these factors impact loan approval decisions.
    """)
    st.write("This table displays the dataset in the first 5 rows")
    st.write(df.head())  # Display the first few rows of the dataset

    st.write("This table displays summary statistics of the dataset.")
    st.write(df.describe())  # Display summary statistics

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)








    # --- EDA ---

    st.markdown("""
                <a name="eda"></a>

                <h2 style="color: #0CABA8;">EDA</h2>

                In this section, we perform Exploratory Data Analysis (EDA) to gain insights into the dataset. We start by examining the distribution of key features and visualizing relationships between them.

                """, unsafe_allow_html=True)


    ### Plot:Distribution of Customer Age ###
    st.write("#### Distribution of Customer Age")

    fig = make_subplots(rows=2, cols=1)

    tr1=go.Box(x=df['customer_age'],name='Age Box Plot',boxmean=True)
    tr2=go.Histogram(x=df['customer_age'],name='Age Histogram')

    fig.add_trace(tr1,row=1,col=1)
    fig.add_trace(tr2,row=2,col=1)

    fig.update_layout(height=700, width=1200)
    st.plotly_chart(fig)

    st.markdown("""
                The customer age shows a normal distribution, meaning that most customers are around a central age, 
                with fewer customers at the younger and older extremes. This indicates a balanced representation of 
                customers across different age groups, making age a reliable feature for prediction.
                """, unsafe_allow_html=True)


    ### Plot:Total Transaction Count by Attrition Flag ###
    st.write("#### Total Transaction Count by Attrition Flag")

    fig = px.histogram(
        df,
        x='total_trans_ct',
        color='attrition_flag',
        labels={
            'total_trans_ct': 'Total Transaction Count',  
            'attrition_flag': 'Attrition Flag',           
            'count': ' Frequency of customers in each transaction count range'                     
        }
    )
    # Display the chart in Streamlit
    st.plotly_chart(fig)
    
    st.markdown("""
                Key Observation:
                Existing Customers (light blue) generally have a higher total transaction count compared to attrited customers, with the distribution peaking around 80 transactions.
                There is a large cluster of existing customers with transaction counts between 40 to 100.
                
                Attrited Customers (darker blue) show a peak in the lower transaction count range, around 40 to 50 transactions.
                The cluster of attrited customers appears more concentrated in the 20 to 60 transaction range, indicating they tend to perform fewer transactions than existing customers.
                
                Overlap: There is an overlap between 20 and 60 transactions, where both existing and attrited customers are present, but attrited customers dominate the lower end of the transaction count.

                Conclusion:
                - This chart suggests that customers with a lower number of transactions are more likely to attrite, while customers with higher transaction counts tend to stay active. Transaction behavior is a significant indicator of customer retention, and customers with fewer transactions may require proactive engagement strategies to prevent attrition.
                """, unsafe_allow_html=True)
    

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)









    # --- Modelling ---

    st.markdown("""
                <a name="modeling"></a>

                <h2 style="color: #0CABA8;">Modelling</h2>

                **Model Performance**
                - Accuracy: The Random Forest model achieved an accuracy of 95.45%, which indicates that 95.45% of the predictions made by the model were correct.

                Precision, Recall, and F1-Score:

                - For class 0 (no attrition), the precision, recall, and F1-score are all high at 0.97, showing that the model performs well in correctly identifying non-attriting customers.
                - For class 1 (attrition), the precision is 0.87, recall is 0.86, and F1-score is 0.86, reflecting slightly lower performance in identifying attriting customers but still reasonably good.
                
                This indicates that the model is highly accurate in predicting customer attrition, but there is still room for improvement, especially in reducing false negatives (customers who attrite but are predicted not to).

                """, unsafe_allow_html=True)

    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)


    fig = px.bar(
        importance_df,
        y='Feature',
        x='Importance',
        title='Feature Importance',
        orientation='h',
        labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
        color='Importance'
    )

    st.plotly_chart(fig)

    st.markdown("""
    The feature importance summary highlights the relative significance of each feature in predicting customer attrition for the bank:

    - Top Features: The most influential features include Total Transaction Count (0.2174), Total Transaction Amount (0.2000), and Total Revolving Balance (0.1016). These are key indicators of customer behavior and seem to have the strongest impact on attrition.

    - Medium Importance: Features like Total Change in Transaction Count (Q4 to Q1) (0.0938), Total Relationship Count (0.0664), and Average Utilization Ratio (0.0662) are moderately important and provide further insights into customer engagement and credit usage.

    - Lower Importance: Demographic and account-related features such as Customer Age (0.0230), Gender (0.0229), and Income Category (0.0216) have less impact, suggesting they play a smaller role in the attrition decision.

    - Least Important: Card Category (0.0015) and features like Dependent Count (0.0089) and Education Level (0.0082) have minimal influence, indicating they may not be as predictive of customer attrition.

    Overall, transaction-related features are the most important, while demographic features are less influential in predicting whether a customer will leave the bank.
    """)





    # --- Conclusion ---

    st.markdown("""
                <a name="conc"></a>

                <h2 style="color: #0CABA8;">Conclusion</h2>

                In this section, we summarize the key findings from the modeling analysis.

                ### Key Findings
                - The Random Forest model demonstrated strong performance with an accuracy of 95.45%, effectively predicting customer attrition.
                - Key features influencing customer attrition include:
                    - `total_trans_ct` (Total Transaction Count), the most important predictor of attrition.
                    - `total_trans_amt` (Total Transaction Amount), another crucial indicator of customer behavior.
                    - `total_revolving_bal`, a significant feature that reflects customers' credit usage.
                    - `credit_limit`, `months_on_book`, and `avg_utilization_ratio` also contribute but to a lesser extent.

                ### Recommendations
                - **Customer Retention**: Based on the model, the bank should focus on customers with low transaction counts and those with high credit balances to improve retention efforts.
                - **Targeted Campaigns**: Develop targeted marketing campaigns for customers identified as high risk (low transaction counts or high revolving balances) to potentially reduce attrition rates.

                Overall, the developed model serves as a valuable tool for predicting customer attrition and can help the bank proactively manage customer relationships.
                """, unsafe_allow_html=True)

    # Add some space before the main content
    st.markdown("<br><br>", unsafe_allow_html=True)