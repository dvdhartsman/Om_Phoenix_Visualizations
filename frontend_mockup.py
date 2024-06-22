import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
import scipy
import joblib


def preprocess_insurance_data(data):
    """
    Preprocessing steps for Insurance_claims_mendeleydata_6.csv
    to be transformed in a manner that allows for state-wise visualization

    Args
    ---------------
    data: pd.DataFrame | pandas dataframe to be used for state-wise plots

    Returns
    ---------------
    data : pd.DataFrame | preprocessed with minimal steps 

    Errors Raised
    ---------------
    KeyError | if data with different column names is used, then this function will raise an error

    """

    data = data.rename(columns={"total_claim_amount":"claim_amount",
                                "insured_sex":"gender"})
    data["gender"] = data["gender"].str.title()

    # Bins for Age Plots
    # bins = [-np.inf, 2, 12, 18, 35, 60, np.inf]
    # labels = ["Infant 0-2", "Child 2-12", "Teenager 12-18", "Young Adult 18-35",
    #       "Adult 35-60", "Senior Citizen 60+"]
    
    # Bins #2
    bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    labels = ["15-20", "20-25", "25-30","30-35", "35-40", "40-45", "45-50", "50-55","55-60",
              "60-65"]
  
    data["age_bracket"] = pd.cut(data["age"], bins=bins, labels=labels)

    data = data.drop(columns=["policy_state", "policy_csl", "policy_deductable", "policy_annual_premium",
                     "umbrella_limit", "policy_number", "capital-gains", "capital-loss", "city", "injury_claim", 
                     "property_claim", "vehicle_claim"])

    data["collision_type"] = data["collision_type"].str.replace("?", "Unattended Vehicle")

    return data


def main():

    df_ins = pd.read_csv("Insurance_claims_mendeleydata_6.csv")

    # Process the insurance Data
    data = preprocess_insurance_data(df_ins)

    model = joblib.load("adaboost_model_insurance_csv.pkl")


    st.header("Car Insurance Model Predictions")

    st.subheader("What can you tell us about your claim?")

    col1, col2 = st.columns(2)

    with col1:
        # Age -------
        age = st.number_input("Age:", min_value = data["age"].min().astype(int), max_value=data["age"].max().astype(int), \
                        value=data["age"].min().astype(int), step=1)

        # gender -----------
        gender_type = st.selectbox("Gender:", [None] + list(data["gender"].unique()),index=0)
        

        # accident_type -----------
        accident_type_type = st.selectbox("Accident Type:", [None] + list(data["accident_type"].unique()),index=0)

        # collision_type -----------
        collision_type = st.selectbox("Collision Type:", [None] + list(data["collision_type"].unique()),index=0)
        

        # incident_severity -----------
        incident_severity_type = st.selectbox("Incident Severity:", [None] + list(data["incident_severity"].unique()),index=0)

    
        # authorities_contacted -----------
        authorities_contacted_type_status = st.selectbox("Authorities Contacted:", [None] + \
                                                            list(data["authorities_contacted"].dropna().unique()),index=0)

    with col2:
        # state -----------
        state_type_status = st.selectbox("State:", [None] + list(data["state"].unique()),index=0)

        # property_damage -----------
        property_damage_type_status = st.selectbox("Property Damage:", [None] + list(data["property_damage"].unique()),index=0)

        # bodily_injuries -----------
        bodily_injuries_type_status = st.selectbox("Number of Bodily Injuries:", [None] + list(data["bodily_injuries"].unique()),index=0)
        
        # police_report_available -----------
        police_report_available_type_status = st.selectbox("Police Report Available?:", [None] + list(data["police_report_available"].unique()),index=0)

        # auto_make -----------
        auto_make_type_status = st.selectbox("Auto Make:", [None] + list(data["auto_make"].unique()),index=0)
        
        # auto_model -----------
        auto_model_type_status = st.selectbox("Auto Model:", [None] + list(data["auto_model"].unique()),index=0)

        # auto_year -----------
        auto_year_type_status = st.selectbox("Auto Year:", [None] + list(data["auto_year"].sort_values(ascending=False).unique()),index=0)
        

    st.write(age, gender_type, accident_type_type, collision_type, incident_severity_type,
             authorities_contacted_type_status, property_damage_type_status, bodily_injuries_type_status,
             police_report_available_type_status, auto_make_type_status, auto_model_type_status,
             auto_year_type_status)

    st.subheader("Generate a Prediction:")
    st.button("Predict Claim:")


if __name__ == "__main__":
    main()