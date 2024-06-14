import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
import scipy


# Code to create the sample dataset with the same number of rows as we have in our modeling set
def preprocess_sample_dataset(df):
    """
    These are the preprocessing steps for sample_data_formatted.csv to be 
    transformed to the format that is used for Statewise Comparison 
    of Car Accident Claims

    Args
    ---------------------
    df_samp: pd.DataFrame | sample dataset to be transformed

    Returns
    ---------------------
    df_samp: pd.DataFrame | dataframe after simple preprocessing

    Errors Raised
    --------------------
    KeyError | If a dataframe is used without the same column names, a KeyError will be raised

    """
    # Get the claim amount as it is final target variable
    df['claim_amount'] = np.where(df['total_bills']<=df['total_coverage'],df['total_bills'],df['total_coverage'])
    # total_bills
    df['total_bills'] = np.where(df['total_bills'].isnull(),df['claim_amount'],df['total_bills'])
    # drop null value rows from claim amount
    df = df.dropna(subset=['claim_amount'])
    # drop rows with claim amount '0'
    df = df[~(df['claim_amount']==0)]
    # drop rows with -ve age
    df = df[~(df['age']<0)]
    
    # String Format for State Abbreviations
    df["state"] = df["state"].str.upper()

    # Selecting States with Fewer Than 45 rows of observations
    small_obs = df["state"].value_counts()[df["state"].value_counts() < 45].index

    # Binning small-observation states
    df.loc[df["state"].isin(small_obs), "state"] = "Other"

    ### Script for Binning Type of Injury Column

    df.rename(columns={"injury_type":"Type of Injury"}, inplace=True)

    df = df.dropna(subset="Type of Injury")
    # First consolidation - the backslash is not separated from 'Other Injury' with a space
    df.loc[df["Type of Injury"] == "Other Injury/ Pain", "Type of Injury"] = "Other Injury / Pain"

    # General Traumatic Brain Injury consolidation
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury.", "Type of Injury"] \
    = "Other Injury / Pain; Traumatic Brain Injury"

    # Removing LOC ("Loss of Consciousness") from the category's granularity
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC", "Type of Injury"] \
    = "Other Injury / Pain; Traumatic Brain Injury"

    # Handling Excessive spaces between words
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic  Brain  Injury", "Type of Injury"] \
    = "Other Injury / Pain; Traumatic Brain Injury"

    # Another Different Entry with Redundant Information
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC; Traumatic  Brain  Injury", "Type of Injury"]\
    = "Other Injury / Pain; Traumatic Brain Injury"

    # Same
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC; Traumatic Brain Injury.", "Type of Injury"]\
    = "Other Injury / Pain; Traumatic Brain Injury"

    # Same
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic  Brain  Injury; Traumatic  Brain  Injury", "Type of Injury"] \
    = "Other Injury / Pain; Traumatic Brain Injury"


    # ---- Handling Fatal Cases ----

    # 124 Distinct Injuries resulting in Death

    # Consolidate entries containing "Death" and "Traumatic Brain Injury"                                   #### CHECK
    df.loc[df["Type of Injury"].str.contains("(?=.*Death)(?=.*Traumatic Brain Injury)"), "Type of Injury"] = "Death"

    # -------- Broken Bones -------
    df.loc[df["Type of Injury"].str.contains("Traumatic Brain Injury.*Broken Bones"), "Type of Injury"]\
    = "Other Injury / Pain; Traumatic Brain Injury; Broken Bones"

    df.loc[df["Type of Injury"].str.contains("Other Injury.*Broken Bones"), "Type of Injury"]\
    = "Other Injury / Pain; Traumatic Brain Injury; Broken Bones"

    # ------- Ruptured Discs -> regular expressions for the "Other Pain" and "Traumatic Brain Injury" as superseding categories
    df.loc[df["Type of Injury"].str.contains("(?=Other Injury / Pain)(?=.*Herniated/Bulging/Ruptured Disc)(?=.*Traumatic Brain Injury)"),\
    "Type of Injury"] = "Other Injury / Pain; Traumatic Brain Injury; Herniated/Bulging/Ruptured Disc"

    # ------ Ruptured Discs -> regular expressions for the "Other Pain" and "Traumatic Brain Injury" as superseding categories
    df.loc[df["Type of Injury"].str.contains("(?=Other Injury / Pain)(?=.*Herniated/Bulging/Ruptured Disc)"),\
    "Type of Injury"] = "Other Injury / Pain; Herniated/Bulging/Ruptured Disc"

    # Capturing the last remaining values
    df.loc[df["Type of Injury"].str.contains("Herniated/Bulging/Ruptured Disc"), "Type of Injury"]\
    = "Other Injury / Pain; Herniated/Bulging/Ruptured Disc"

    ##### AT THIS POINT: Remaining un-consolidated values all represent less than 1% of total values #########

    # -------- Tendon/Ligament -> Consolidating into the larger bin
    df.loc[df["Type of Injury"] == "Tendon/Ligament Tear/Rupture", "Type of Injury"] = "Other Injury / Pain; Tendon/Ligament Tear/Rupture"

    # ------------PTSD etc using the larger bin in existence
    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury - w/LOC; Anxiety,PTSD,Depression,Stress", "Type of Injury"]\
    = "Other Injury / Pain; Traumatic  Brain  Injury; Anxiety,PTSD,Depression,Stress"

    df.loc[df["Type of Injury"] == "Other Injury / Pain; Traumatic Brain Injury.; Anxiety,PTSD,Depression,Stress", "Type of Injury"]\
    = "Other Injury / Pain; Traumatic  Brain  Injury; Anxiety,PTSD,Depression,Stress"

    # After these bins have been created, the top 20 Values account for 96.2% of all rows in the data.

    # Binning all values not found in the top 20
    exclusion_list = df["Type of Injury"].value_counts(normalize =True)[:20].index

    # '~' accesses the complement - "Not In" the exclusion list
    df.loc[~df["Type of Injury"].isin(exclusion_list), "Type of Injury"] = "Other Injury"

    # TBI
    df.loc[df["Type of Injury"].str.contains("Traumatic Brain Injury"), "Type of Injury"] = "Traumatic Brain Injury"

    # Broken Bones
    df.loc[df["Type of Injury"].str.contains("Broken Bones"), "Type of Injury"] = "Broken Bones"

    # 21 Bins of Values left, and the final bin contains roughly 3.8 % of all entries
    df["Type of Injury"] = df["Type of Injury"].str.replace("Other Injury / ", "")

    df["Type of Injury"] = df["Type of Injury"].str.replace("Pain; ", "")

    # Bins for Age Plots
    bins = [-np.inf, 2, 12, 18, 35, 60, np.inf]
    labels = ["Infant 0-2", "Child 2-12", "Teenager 12-18", "Young Adult 18-35",
          "Adult 35-60", "Senior Citizen 60+"]
    
    df["age_bracket"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Filling Nulls Logically
    df["airbag_deployed"] = df["airbag_deployed"].fillna("Unknown")

    df["accident_type"] = df["accident_type"].str.replace("It involved multiple cars", "Multi Car")
    df["accident_type"] = df["accident_type"].fillna("Unknown")
    
    # dropping this subset because it would impede our ability to filter data at the end
    df = df.dropna(subset="age")

    # Remove <0 values from "claim_amount" only impacts values of -2
    df.loc[df["claim_amount"] < 0, "claim_amount"] = 0

    # drop cities
    df = df.drop(columns=["city", "other_injury", "serious_injury", "potential_tbi"])

    return df


# Processing for insurance data
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
    bins = [-np.inf, 2, 12, 18, 35, 60, np.inf]
    labels = ["Infant 0-2", "Child 2-12", "Teenager 12-18", "Young Adult 18-35",
          "Adult 35-60", "Senior Citizen 60+"]
  
    data["age_bracket"] = pd.cut(data["age"], bins=bins, labels=labels)

    return data


# Statewise Plots -------------------------------------------

def plotly_states(data):
    """
    Function to generate a plotly figure of barplots of mean and median state claim values for car accidents
    compatible with sample_data_formatted.csv and Insurance_claims_mendeleydata_6.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["state", "total_claim_amount"]

    Returns
    -----------
    plotly figure | barplot with hover values of State, Mean/Median Value 

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """

    # Filtering out miscellaneous states
    data = data[data["state"] != "Other"]

    # Grouping data by state and calculating median and mean
    grouped = data.groupby("state")["claim_amount"].agg(["median", "mean"]).sort_values(by="median", ascending=False)

    # Resetting index to make 'state' a column for Plotly
    grouped = grouped.reset_index()

    # Creating Plotly figure
    fig = px.bar(grouped, x='state', y=['median', 'mean'],
                 labels={'value': 'Claim Amount in USD', 'state': 'States'},
                 title='Mean and Median Claims by State Sorted by Median Claim',
                 barmode='group', 
                 template="plotly")
    
    # Legend
    fig.update_layout(legend_title='')

    # Customizing hover info
    fig.update_traces(hovertemplate='State: %{x}<br>Value: %{y:.2f}')

    fig.for_each_trace(lambda t: t.update(name=t.name.capitalize()))

    # Returning the Plotly figure
    return fig


# Boxplots for State Car Accident Claim Distributions 

def plotly_box_states(data):
    """
    Function to generate a plotly figure of boxplots of car accidents claim distributions by state
    compatible with sample_data_formatted.csv and Insurance_claims_mendeleydata_6.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["state", "claim_amount"]

    Returns
    -----------
    plotly figure | boxplot with hover values of State, [min, lower fence, 25 percentile, median, 75 percentile, upper fence, max] 

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """
    
    # Filter Data for States == Other
    data = data[data["state"] != "Other"]

    # Creating a list of states ordered by their median percentile value 
    # to provide a left-to-right visual structure 
    upper_q = list(data.groupby("state")["claim_amount"].median().sort_values(ascending=False).index)
    
    # Create traces for each state -> this was the only way I could get the whisker/plot scale correct
    traces = []
    for state in upper_q:
        state_data = data[data['state'] == state]
        trace = go.Box(
            y=state_data['claim_amount'],
            name=state,
            boxpoints='all',  # Show all points to maintain correct whisker length
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(opacity=0),  # Make point markers invisible
            line=dict(width=2),
            boxmean=False  # Do not show mean
        )
        traces.append(trace)

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title="Distribution of Car Accident Claims in Different States",
        yaxis=dict(
            title="Total Claim in USD"
        ),
        xaxis=dict(
            title="State"
        ),
        showlegend=False,
        template="plotly"
    )
    
    # Calculate IQR for each state to determine y-axis range
    iqr_ranges = data.groupby('state')['claim_amount'].apply(lambda x: (x.quantile(0.25), x.quantile(0.75)))
    iqr_min, iqr_max = iqr_ranges.apply(lambda x: x[0]).min(), iqr_ranges.apply(lambda x: x[1]).max()
    iqr = iqr_max - iqr_min

    # Update y-axis range to be slightly larger than the IQR range
    fig.update_yaxes(range=[-1000, iqr_max + 1.5 * iqr])

    return fig


# Gender Plots -----------------------------------------------------

# Plot to show KDEs of male, female and overlay of both
def plotly_gender(data):
    """
    Function to generate a plotly figure of KDE distributions for Genders 
    compatible with Kaggle_medical_practice_20.csv and Insurance_claims_mendeleydata_6.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["gender", "total_claim_amount"]

    Returns
    -----------
    plotly figure | 3 kde plots with hover values of x coordinates (claim value)

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """
    if "claim_amount" in data.columns:
        data = data.rename(columns={"claim_amount":"total_claim_amount"})

    male_data = data.query("gender == 'Male'")['total_claim_amount']
    female_data = data.query("gender == 'Female'")['total_claim_amount']

    male_median_x = male_data.median()
    female_median_x = female_data.median()

    # KDEs
    male_kde = ff.create_distplot([male_data], group_labels=['Male'], show_hist=False, show_rug=False)
    female_kde = ff.create_distplot([female_data], group_labels=['Female'], show_hist=False, show_rug=False)

    # Create subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=(f'Male Claim Amounts - Median Claim ${male_median_x:,.2f}', f'Female Claim Amounts - Median Claim ${female_median_x:,.2f}', 'Male vs Women Overlaid'),
                        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"colspan": 2}, None]],
                        row_heights=[0.5, 0.5],
                        column_widths=[0.5, 0.5])

    # Male KDE Plot
    for trace in male_kde['data']:
        trace["hoverinfo"] = 'x'
        trace["showlegend"] = False
        fig.add_trace(trace, row=1, col=1)

    male_median_y = male_kde['data'][0]['y'].max()
    
    # Adding a vline
    fig.add_shape(type="line",
                  x0=male_median_x, y0=0,
                  x1=male_median_x, y1=male_median_y,
                  line={"color":"darkred","dash":"dash"},
                  row=1, col=1)

    # Female KDE Plot
    for trace in female_kde['data']:
        trace["hoverinfo"] = 'x'
        trace["showlegend"] = False
        fig.add_trace(trace, row=1, col=2)

    female_median_y = female_kde['data'][0]['y'].max()
    
    # Adding a vline
    fig.add_shape(type="line", 
              x0=female_median_x, y0=0, 
              x1=female_median_x, y1=female_median_y, 
              line=dict(color="darkred", dash="dash"),
              row=1, col=2)

    # Overlaid KDE Plot
    fig.add_trace(go.Scatter(x=male_kde['data'][0]['x'], y=male_kde['data'][0]['y'], 
                             mode='lines', name='Male', fill='tozeroy', line=dict(color='blue'), opacity=0.1,
                             hoverinfo='x'), 
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=female_kde['data'][0]['x'], y=female_kde['data'][0]['y'], 
                             mode='lines', name='Female', fill='tozeroy', line=dict(color='lightcoral'), opacity=0.1,
                             hoverinfo='x'), 
                  row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1200, title_text="Distribution of All Claim Amounts for Men vs Women")
    fig.update_xaxes(title_text="Total Claim in USD", row=1, col=1)
    fig.update_xaxes(title_text="Total Claim in USD", row=1, col=2)
    fig.update_xaxes(title_text="Total Claim in USD", row=2, col=1)

    fig.update_layout(showlegend=True, legend=dict(x=0.875, y=0.275))
    fig.update_yaxes(showticklabels=False)

    # Show plot
    return fig


def plotly_box_gender(data):
    """
    Function to generate a plotly figure of Boxplot distributions without outliers for Genders Across Insurance Types
    compatible with Kaggle_medical_practice_20.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["gender", "total_claim_amount", "insurance"]

    Returns
    -----------
    plotly figure | boxplot with hover values of State, then any of: 
    [max, upper fence, 75th percentile, median, 25th percentile, lower fence, min]

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """

    # Creating a list of states ordered by their median percentile value 
    # to provide a left-to-right visual structure 
    fig = px.box(data, x="insurance", y="total_claim_amount", color="gender")

    # Update layout
    fig.update_layout(
        title="Distribution of Claims by Insurance Type for Men and Women",
        yaxis=dict(
            title="Total Claim in USD"
        ),
        xaxis=dict(
            title="Insurance Type"
        ),
        showlegend=True,
        template="plotly"
    )
    
    fig.update_layout(legend_title='Gender')
    
    return fig


    # Plot for different types of injuries from the Sample Data
def plotly_injury_bar(data):
    """
    Compatible with Sample Dataset
    """
    grouped = data.groupby("Type of Injury")["claim_amount"].agg(["mean", "median"]).round(2).reset_index().sort_values(by="median", ascending=True).rename(columns={"mean":"Mean", "median":"Median"})
    fig = px.bar(grouped, y='Type of Injury', x=['Median', 'Mean'],
             labels={'value': "Claim Value", 'Type of Injury': 'Injury', "variable":"Statistic"},
             title='Mean and Median Claims by Injury', barmode='group', color_continuous_scale="Viridis")
    fig.update_layout(showlegend=True, width=1200, height=675)

    return fig


    # Histplot function for injuries
def plotly_injury_hist(data):
    fig_h = px.histogram(data, x="claim_amount", nbins=25, labels={"claim_amount":"Claim", "value":"Count"})
    fig_h.update_traces(hovertemplate='Claim: %{x}<br>Count: %{y}')
    injury = data["Type of Injury"].unique()[0]
    fig_h.update_layout(yaxis={"title":"Count"}, title=f"Histogram of Claim Distribution for {injury.title()}")
    return fig_h

# Boxplot for injuries
def plotly_boxplot_injury(data):
    fig_b = px.box(data, x="claim_amount", labels={"claim_amount":"Claim"})
    injury = data["Type of Injury"].unique()[0]
    fig_b.update_layout(title=f"Boxplot of Claim Distribution for {injury.title()}")

    return fig_b

# AGE ------------------------
def plotly_age(data):
    age_data = data.dropna(subset="age")
    age_data["age"] = age_data["age"].astype("int8")
    age_data = age_data.sort_values(by="age", ascending = True)
    
    fig = px.line(age_data.groupby("age")["claim_amount"].agg(["median"])\
                  .round(-2).reset_index(), x="age", y="median", \
                  labels={"median":"Median Claim", "age":"Age"}, title="Median Claim Value by Age")
    fig.update_traces(name="Median Claim Value", showlegend=True)
    fig.update_layout(legend_title="")
    
    return fig

def plotly_age_hist(data):
    fig = px.histogram(data["age"], labels={"age":"Age"}, title="Total # of Claims by Age", 
                  color_discrete_sequence=["orange"])
    fig.update_layout(legend_title="", xaxis={"title":"Age"}, yaxis={"title":"Number of Claims"})
    fig.update_traces(name="Claims")
    
    return fig

def plotly_age_counts(data):
    vcounts = data["age"].value_counts().sort_index()
    fig = px.line(vcounts, labels={"age":"Age", "value":"Number of Claims"}, title="Total # of Claims by Age")
    fig.update_layout(legend_title="")
    fig.update_traces(name="Claims")
    
    return fig


def plotly_age_bracket(data):
    group = data.groupby("age_bracket")["claim_amount"].agg(["median", "mean"]).round(-2)\
    .rename(columns={"median":"Median", "mean":"Mean"})
    fig = px.bar(group.reset_index(), y="age_bracket", x=["Median", "Mean"], 
                 title="Mean and Median Claims by Age Bracket",\
                 labels={"age_bracket":"Group", "median":"Median", "mean":"Mean"},
                barmode="group", color_discrete_sequence=["red", "royalblue"])

    fig.update_layout(legend_title_text="Statistic")
    fig.update_traces(hovertemplate="Claim Amount: %{x} <br>Group: %{y}")
    
    return fig


def plotly_scatter_age(data, group=None):
    fig = px.scatter(data, x="age", y="claim_amount", log_y=False, range_y=[0, data["claim_amount"].max()],
                     title="Claim Value vs Age (Zoom to Inspect, Click Legend to Activate/Deactivate Groups)", color=group, symbol=group)
    fig.update_layout(xaxis={"title":"Age"}, yaxis={"title":"Claim Value"})

    return fig

# ----------------- Plots for Filtered Data

def plotly_filtered_claims(data, condition):
    fig = px.histogram(data["claim_amount"], labels={"claim_amount":"Claim Value USD"}, title=f"Number of Claims by Claim Value - {condition}", 
                  color_discrete_sequence=["blue"], nbins=20)
    fig.update_layout(legend_title="", xaxis={"title":"Claim Value"}, yaxis={"title":"Number of Claims"})
    fig.update_traces(name="Claims", marker_line_color='black', marker_line_width=1.5)
    
    return fig

# Boxplot for injuries
def plotly_boxplot_filtered(data, condition):
    fig_b = px.box(data, x="claim_amount", labels={"claim_amount":"Claim"})
    fig_b.update_layout(title=f"Boxplot of Claim Distribution for {condition}")

    return fig_b



# Define the function plotly_filtered_claims
def plotly_filtered_claims_bar(filtered_data, excluded_data, original_data):
    filtered_med = filtered_data["claim_amount"].median()
    filtered_mean = filtered_data["claim_amount"].mean()
    
    excluded_med = excluded_data["claim_amount"].median()
    excluded_mean = excluded_data["claim_amount"].mean()

    original_med = original_data["claim_amount"].median()
    original_mean =  original_data["claim_amount"].mean()
    fig1 = px.bar(x=["Selected Data", "Excluded Data", "All Data"," ", "Selected Mean Claim", "Excluded Mean Claim", "All Data Mean Claim"], 
                  y=[filtered_med, excluded_med, original_med, 0, filtered_mean, excluded_mean, original_mean], 
                  color=["Selected Median Claim", "Excluded Median Claim", "Original Median Claim","",  "Selected Mean Claim","Excluded Mean Claim", "Original Mean Claim"], 
                  color_discrete_sequence=["lightgreen", "red","blue", "red", "lightgreen", "red", "blue"],
                labels={"red":"Filtered Median"})
    
    # fig1.add_trace(
    fig1.update_layout(showlegend=True, title="Comparison of Mean and Median Values", 
                       xaxis={"title":"Data/Statistic"}, yaxis={"title":"Claim Amount USD"})
    fig1.update_layout(legend_title="Dataset/Statistic")
    
    return fig1


def plotly_filtered_claims_bar(data):
    fig = px.bar(
    data[data["Statistic"].isin(["Average Value", "Median Value"])],
    x = "Statistic", y=["Selected Data", "Excluded Data", "All Data"],
    barmode="group", template="plotly_dark",
    title="Comparison of Average and Median Claim Values")

    fig.update_layout(legend_title="Dataset", yaxis={"title":"Claim Value USD"}, bargap=.35)                      
    fig.update_traces(hovertemplate="Claim Amount %{y}<br> Statistic: %{x}<br>")

    return fig
    
# ----- main function ------------------------------------------------------------------

def main():
    # Sample Data
    df_sample = pd.read_csv("sample_data_formatted.csv")

    # Medical Practice Data
    df_med = pd.read_csv("Kaggle_medical_practice_20.csv", index_col=0)

    # Third Data
    df_ins = pd.read_csv("Insurance_claims_mendeleydata_6.csv")

    # Process the sample data
    df_sample = preprocess_sample_dataset(df_sample)

    # Process the insurance Data
    df_ins = preprocess_insurance_data(df_ins)

    st.header("Choose one of our 3 datasets")
    data_source = st.selectbox("Choose Data", ["Car Accident Claims (Sample Data)", "Medical Malpractice", "Auto Insurance Claims"])

    if data_source == "Car Accident Claims (Sample Data)":
        data = df_sample
        st.subheader("This dataset is comprised of car accident claims.")
        st.markdown("---")
        st.subheader("State-wise Data")
        st.plotly_chart(plotly_states(data))
        st.plotly_chart(plotly_box_states(data))
        st.header("Injury Type")

        # Bar Chart of Injury Types
        st.plotly_chart(plotly_injury_bar(data))

        # Option to pursue individual injuries
        injury = st.selectbox("Choose an Injury to See the Distribution", data["Type of Injury"].unique())
        inj_data = data.loc[(data["Type of Injury"] == injury) & (data["claim_amount"] < data["claim_amount"].quantile(.95))]
        

        # Injury Type Plots for Sample Data
        st.header("Distributions of Claims by Injury")

        # Histogram
        st.subheader("Histogram")
        st.plotly_chart(plotly_injury_hist(inj_data))

        # Boxplot
        st.subheader("Boxplot")
        st.plotly_chart(plotly_boxplot_injury(inj_data))

        # ----------------- AGE ----------------
        # Histogram
        st.subheader("Age:")
        st.plotly_chart(plotly_age_hist(data))
        
        # Line plot of Age Value Counts - Like a Histogram
        st.plotly_chart(plotly_age_counts(data))
        
        # Scatterplot of Claim vs Age
        keys = [None] + list(data.select_dtypes(object).columns.str.title().str.replace("_", " "))
        values = [None] + list(data.select_dtypes(object).columns)
        age_col_dict = dict(zip(keys, values))

        group = st.selectbox("Add Detail for Subsets:", keys)
        st.plotly_chart(plotly_scatter_age(data, age_col_dict[group]))
        

        # Line Plot of Median Claims by Age
        st.plotly_chart(plotly_age(data))

        # Mean and Median Barplots
        st.plotly_chart(plotly_age_bracket(data))

        # Multi-filterable Plot for Sample Dataset
        st.header("Try Out Multiple Filters:")
        st.write('If you would like to deactivate a filter select: "None"')
        
        # Age -------
        min_age, max_age = st.slider("Age Range", min_value = data["age"].min().astype(int), max_value=data["age"].max().astype(int), \
                        value=(data["age"].min().astype(int), data["age"].max().astype(int)), step=1)
        
        # Boolean Mask for the Filter
        age_condition = (data["age"] >= min_age) & (data["age"] <= max_age)
        

        # Accident Type ----------------------
        accident_type_status = st.selectbox("Type of Accident:", [None] + list(data["accident_type"].unique()),index=0)
        if accident_type_status:
            accident_type_condition = (data["accident_type"] == accident_type_status)
        else:
            accident_type_condition = True

        
        # Airbag Deployed? -------------------
        airbag_status = st.selectbox("Airbag Deployment:", 
                                   [None] + list(data["airbag_deployed"].unique()),index=0)
        if airbag_status:
            airbag_condition = (data["airbag_deployed"] == airbag_status)
        else:
            airbag_condition = True


        # Truck or Bus Involved -----------------------------
        truck_involved_status = st.selectbox("Truck or Bus Involved?:", [None] + list(data["truck_bus_involved"].dropna().unique()),index=0)
        if truck_involved_status:
            truck_involved_condition = (data["truck_bus_involved"] == truck_involved_status)
        else:
            truck_involved_condition = True

        # Taxi Involved -----------------------------
        taxi_involved_status = st.selectbox("Taxi Involved?:", [None] + list(data["taxi_involved"].dropna().unique()),index=0)
        if taxi_involved_status:
            taxi_involved_condition = (data["taxi_involved"] == taxi_involved_status)
        else:
            taxi_involved_condition = True


        # Called 911 After -----------------------------
        called911_status = st.selectbox("Did You Call 911?:", [None] + list(data["called_911"].dropna().unique()),index=0)
        if called911_status:
            called911_condition = (data["called_911"] == called911_status)
        else:
            called911_condition = True

        
        # Type of Injury -----------
        injury_type = st.selectbox("Type of Injury:", 
                                   [None] + list(data["Type of Injury"].unique()),index=0)

        if injury_type:
            injury_condition = (data["Type of Injury"] == injury_type)
        else:
            injury_condition = True
            
        
        # positive_mri_finding -----------------------------
        mri_status = st.selectbox("MRI Positive?:", [None] + list(data["positive_mri_finding"].dropna().unique()),index=0)
        if mri_status:
            mri_condition = (data["positive_mri_finding"] == mri_status)
        else:
            mri_condition = True


        # surgery_injection_recom -----------------------------
        surgery_status = st.selectbox("Surgery or Injections Recommended?:", [None] + list(data["surgery_injection_recom"].dropna().unique()),index=0)
        if surgery_status:
            surgery_condition = (data["surgery_injection_recom"] == surgery_status)
        else:
            surgery_condition = True

        
        #------------ Apply all of the filters --------------------
        all_conditions = age_condition & injury_condition & airbag_condition & accident_type_condition & truck_involved_condition & taxi_involved_condition & called911_condition & mri_condition & surgery_condition

        st.markdown("---")
        st.write("Here's a brief summary of claims for the filters you have selected:")
        
        # DF for comparison of numeric profiles
        description_table = pd.DataFrame(data[all_conditions]["claim_amount"].describe().round(2)).reset_index()\
                                .merge(pd.DataFrame(data[~all_conditions]["claim_amount"].describe()).reset_index()\
                                .rename(columns={"claim_amount":"Excluded Data"})\
                                .round(2)).reset_index()\
                                .rename(columns={
                                    "claim_amount":"Selected Data",
                                    "index":"Statistic"}).drop(columns="level_0")
        
        # DF of all rows description
        describe_df = pd.DataFrame()
        describe_df["Statistic"] = description_table["Statistic"].copy()
        describe_df["All Data"] = data["claim_amount"].describe().values.round(2)

        # Merge 3rd df
        description_table = description_table.merge(describe_df, on="Statistic")

        # Mapping the statistic values
        description_table["Statistic"] = description_table["Statistic"].map({"count":"Number of Rows",
                   "mean":"Average Value",
                   "std":"Standard Deviation",
                   "min":"Minimum Value",
                   "25%":"25th Percentile Value",
                   "50%":"Median Value",
                   "75%":"75th Percentile Value",
                   "max":"Maximum Value"})
        
        description_table.drop(2, inplace=True)
        description_table = description_table.iloc[[0,2,3,4,1,5,6], :]

        # Sample size warning
        if data[all_conditions].shape[0] <= 10:
            st.write("This is a small subset of data, so use discretion when interpretting the results.")

        # Display the dataframe
        st.dataframe(description_table, use_container_width=True, hide_index=True)

        # Only display distribution plots if there are 10 or more observations
        if data[all_conditions].shape[0] >= 10:
            distribution_skew_condition = (data[all_conditions]["claim_amount"].max() - data[all_conditions]["claim_amount"].quantile(.9)) >\
            (data[all_conditions]["claim_amount"].quantile(.9) - data[all_conditions]["claim_amount"].quantile(.75))
            
            # Account for extreme outliers
            if distribution_skew_condition:
                hist_data = data[all_conditions][data[all_conditions]["claim_amount"]\
                                                  < data[all_conditions]["claim_amount"].quantile(.9)]
            
                condition = "Selected Data without Extreme Outliers"
                # Histogram
                st.plotly_chart(plotly_filtered_claims(hist_data, condition))
                # Boxplot
                st.plotly_chart(plotly_boxplot_filtered(hist_data, condition))

            
            else: # If not distribution_skew_condition
                condition = "Selected Data"
                st.plotly_chart(plotly_filtered_claims(data[all_conditions], condition))
                # Boxplot
                st.plotly_chart(plotly_boxplot_filtered(data[all_conditions], condition))

        # Comparison Bar Plot
        # st.plotly_chart(plotly_filtered_claims_bar(data[all_conditions],data[~all_conditions], data))
        st.plotly_chart(plotly_filtered_claims_bar(description_table))
        


    # MEDICAL PRACTICE ------------------------------------------------------------------
    elif data_source == "Medical Malpractice": 
        data = df_med
        st.subheader("This dataset is comprised of medical malpractice claims.")
        st.markdown("---")
        st.header("Gender Data")
        st.plotly_chart(plotly_gender(data))
        st.subheader("Gender and Insurance Types")
        st.plotly_chart(plotly_box_gender(data))
    
    elif data_source == "Auto Insurance Claims":
        data = df_ins
        st.subheader("This dataset is comprised of car accident claims.")
        st.markdown("---")
        st.subheader("State-wise Data")
        st.plotly_chart(plotly_states(data))
        st.plotly_chart(plotly_box_states(data))
        st.subheader("Gender Data")
        st.plotly_chart(plotly_gender(data))



if __name__ == "__main__":
    main()

    
