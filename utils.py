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

    #------------ FROM MARIAMS CODE ---------------------------
    df['airbag_deployed'] = df['airbag_deployed'].fillna('No')

    df['called_911'] = df['called_911'].fillna('Unknown')
    
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
    fig.update_traces(hovertemplate='State: %{x}<br>Value: %{y:$,.2f}')

    fig.for_each_trace(lambda t: t.update(name=t.name.capitalize()))
    fig.update_layout(yaxis=dict(tickformat='$,.2f'))
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

def plotly_gender(data):
    """
    Function to generate a plotly figure of KDE distributions for Genders 
    compatible with Kaggle_medical_practice_20.csv and Insurance_claims_mendeleydata_6.csv

    Args
    -----------
    data: pd.DataFrame | data with columns: ["gender", "total_claim_amount"]

    Returns
    -----------
    plotly figure | kde plots overlaid with hover values of x coordinates (claim value)

    Errors
    -----------
    KeyError if data do not contain the correct columns
    """
    male_data = data.query("gender == 'Male'")['claim_amount']
    female_data = data.query("gender == 'Female'")['claim_amount']

    male_median_x = male_data.median().round(2)
    female_median_x = female_data.median().round(2)  

    male_kde = ff.create_distplot([male_data], group_labels=['Male'], show_hist=False, show_rug=False)
    female_kde = ff.create_distplot([female_data], group_labels=['Female'], show_hist=False, show_rug=False)

    # Create the overlaid plot
    fig = go.Figure()

    # Male KDE Plot
    fig.add_trace(go.Scatter(x=male_kde['data'][0]['x'], y=male_kde['data'][0]['y'], 
                             mode='lines', name='Male', fill='tozeroy', line=dict(color='blue'), opacity=0.1,
                             hoverinfo='x', xhoverformat="$,.2f", hovertemplate='Claim Amount: %{x:$,.2f}'))

    # Female KDE Plot
    fig.add_trace(go.Scatter(x=female_kde['data'][0]['x'], y=female_kde['data'][0]['y'], 
                             mode='lines', name='Female', fill='tozeroy', line=dict(color='lightcoral'), opacity=0.1,
                             hoverinfo='x', xhoverformat="$,.2f", hovertemplate='Claim Amount: %{x:$,.2f}'))

    # Adding vertical lines for medians as scatter traces for legend
    male_median_y = max(male_kde['data'][0]['y'])
    female_median_y = max(female_kde['data'][0]['y'])

    fig.add_trace(go.Scatter(
        x=[male_median_x, male_median_x], y=[0, male_median_y],
        mode="lines",
        line=dict(color="lightblue", dash="dash"),
        name=f"Male Median ${male_median_x:,.0f}"
    ))

    fig.add_trace(go.Scatter(
        x=[female_median_x, female_median_x], y=[0, female_median_y],
        mode="lines",
        line=dict(color="lightpink", dash="dash"),
        name=f"Female Median ${female_median_x:,.0f}"
    ))

    # Update layout
    fig.update_layout(height=600, width=800, 
                      title_text="Claim Distribution - Men vs Women: Higher Peaks Indicate More-Common Claim Amounts",
                      xaxis_title="Total Claim in USD",
                      yaxis_title="Density",
                      showlegend=True,
                      legend=dict(x=0.825, y=0.875))
    fig.update_yaxes(showticklabels=False)

    return fig


# fig.update_layout(yaxis=dict(tickformat='$,.2f'))

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
    fig = px.box(data, x="insurance", y="claim_amount", color="gender")

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
def plotly_injury_bar(data, group, **kwargs):
    """
    Compatible with Sample Dataset, inverts x and y 
    """
    grouped =  data.groupby(group)["claim_amount"].agg(["mean", "median"]).round(2).reset_index().sort_values(by="median", ascending=True).rename(columns={"mean":"Mean", "median":"Median"})
    fig = px.bar(data_frame = grouped, y=group, x=['Median', 'Mean'],
             labels={'value': "Claim Value", group:group.replace("_", " ").title(), "variable":"Statistic"},
             title=f'Mean and Median Claims by {group.replace("_", " ").title()}', barmode='group', **kwargs)
    fig.update_layout(showlegend=True, width=1200, height=675)
    fig.update_layout(xaxis=dict(tickformat='$,.2f'))

    return fig


    # Histplot function for injuries
def plotly_injury_hist(data):
    fig_h = px.histogram(data, x="claim_amount", nbins=25, labels={"claim_amount":"Claim", "value":"Count"})
    fig_h.update_traces(hovertemplate='Claim: %{x}<br>Count: %{y}')
    injury = data["Type of Injury"].unique()[0]
    fig_h.update_layout(yaxis={"title":"Count"}, title=f"Histogram of Claim Distribution for {injury.title()}")
    fig_h.update_traces(name="Claims", marker_line_color='black', marker_line_width=1.5)
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
    fig.update_layout(yaxis=dict(tickformat='$,.2f'))
    
    return fig

def plotly_age_hist(data, **kwargs):
    fig = px.histogram(data_frame=data["age"], labels={"age":"Age"}, title="Number of Claims by Age", **kwargs)
    fig.update_layout(legend_title="", xaxis={"title":"Age"}, yaxis={"title":"Number of Claims"}, showlegend=False)
    fig.update_traces(name="Claims", hovertemplate="Age %{x}<br> Number of Claims %{y}")
    fig.update_traces(name="Claims", marker_line_color='black', marker_line_width=1.5)
    
    return fig

def plotly_age_counts(data):
    vcounts = data["age"].value_counts().sort_index()
    fig = px.line(vcounts, labels={"age":"Age", "value":"Number of Claims"}, title="Total # of Claims by Age")
    fig.update_layout(legend_title="")
    fig.update_traces(name="Claims")
    
    return fig


def plotly_age_bracket(data, **kwargs):
    group = data.groupby("age_bracket")["claim_amount"].agg(["median", "mean"]).round(-2).sort_index(ascending=False)\
    .rename(columns={"median":"Median", "mean":"Mean"})
    
    fig = px.bar(data_frame=group.reset_index(), y="age_bracket", x=["Median", "Mean"], 
                 title="Mean and Median Claims by Age Bracket",\
                 labels={"age_bracket":"Age Group", "median":"Median", "mean":"Mean"},
                barmode="group", **kwargs)

    fig.update_layout(legend_title_text="Statistic")
    fig.update_traces(hovertemplate="Claim Amount: %{x} <br>Age Group: %{y}")
    fig.update_layout(xaxis=dict(tickformat='$,.2f', title="Claim Amount"), yaxis=dict(title="Age Group"))
    
    return fig

def plotly_age_line(data, group, **kwargs):
    grouped = data.groupby(group)["claim_amount"].agg(["median", "mean"]).round(-2).sort_index()\
    .rename(columns={"median":"Median", "mean":"Mean"}).reset_index()
    fig = px.line(data_frame = grouped, x=group, y=["Median","Mean"], 
                title=f"Trends in Claim Values Across {group.replace('_', ' ').title()}",
                labels=dict(group=group.replace("_", " ").title(), median="Median", mean="Mean"), markers=True, **kwargs)
    fig.update_layout(legend_title_text="Statistic")
    fig.update_traces(hovertemplate="Claim Amount: %{x} <br>Group: %{y}")
    fig.update_layout(yaxis=dict(tickformat='$,.2f', title="Claim Amount"), xaxis=dict(title=group.replace('_', ' ').title()))

    return fig


def plotly_scatter_age(data, group=None):
    
    fig = px.scatter(data, x="age", y="claim_amount", log_y=False, range_y=[0, data["claim_amount"].max()],
                     title="Claim Value vs Age (Zoom to Inspect, Click Legend to Activate/Deactivate Groups)",
                       color=group, symbol=group, 
                       labels={group:group.replace("_", " ").title() if group else group, 
                               "age":"Age", "claim_amount":"Claim Amount"})
    leg_title = group.replace('_', ' ').title() if group is not None else group
    fig.update_layout(xaxis={"title":"Age"}, yaxis={"title":"Claim Value"},
                      legend_title=f"{leg_title}")
    fig.update_layout(scattermode="group", scattergap=.75)
    fig.update_layout(yaxis=dict(tickformat='$,.2f'))
    
    
                                                

    return fig


def plotly_pie(data, column, **kwargs):
    fig = px.pie(data_frame=data, names=column, hole=.5, 
                 title=f"Proportions Observed in the Data: {column.replace('_', ' ').title()}", 
                 labels={column:column.replace('_', ' ').title()}, **kwargs)
    fig.update_layout(legend_title_text = f"{column.replace('_', ' ').title()}")
    # fig.update_traces(hovertemplate=f"Claim Amount %{y}<br> Statistic: %{x}<br>")
    return fig


# ----------------------- Mariam Functions -------------------------------
def plotly_mean_median_bar(data, group, **kwargs): # KWARGS --------
    """
    Compatible with Most Datasets 
    """
    if "total_claim_amount" in data.columns:
        data = data.rename(columns={"total_claim_amount":"claim_amount"})
    grouped = data.groupby(group)["claim_amount"].agg(["mean", "median"]).round(2).reset_index().sort_values(by="median", ascending=True).rename(columns={"mean":"Mean", "median":"Median"})
    fig = px.bar(data_frame=grouped, x=group, y=['Median', 'Mean'],
             labels={'value': "Claim Value", group:group.replace("_", " ").title(), "variable":"Statistic"},
             title=f'Mean and Median Claims by {group.replace("_", " ").title()}', barmode='group', 
             color_continuous_scale="Viridis", **kwargs)  # KWARGS!!!!!!!!!!
    fig.update_layout(showlegend=True, width=1200, height=675)
    fig.update_layout(yaxis=dict(tickformat='$,.2f'))

    return fig


# ----------------- Plots for Filtered Data

def plotly_filtered_claims(data, condition, **kwargs):
    fig = px.histogram(data_frame=data["claim_amount"], labels={"claim_amount":"Claim Value USD"}, 
                       title=f"Number of Claims by Value - {condition}", nbins=20, **kwargs)
    fig.update_layout(legend_title="", xaxis={"title":"Claim Value"}, yaxis={"title":"Number of Claims"})
    fig.update_traces(name="Claims", marker_line_color='black', marker_line_width=1.5,
                      hovertemplate="Claim Value: %{x}<br> Number of Claims: %{y}")
    fig.update_layout(xaxis=dict(tickformat='$,.2f'),showlegend=False)
    
    return fig

# Boxplot for filtered data
def plotly_boxplot_filtered(data, condition, **kwargs):
    fig_b = px.box(data_frame= data, x="claim_amount", labels={"claim_amount":"Claim"}, **kwargs)
    fig_b.update_layout(title=f"Boxplot of Claim Distribution for {condition}")
    fig_b.update_layout(xaxis=dict(tickformat='$,.2f'))

    return fig_b


def plotly_filtered_claims_bar(data, **kwargs):
    fig = px.bar(data_frame = data[data["Statistic"].isin(["Average Value", "Median Value"])],
    x = "Statistic", y=["Selected Data", "Excluded Data", "All Data"],
    barmode="group",
    title="Comparison of Average and Median Claim Values", **kwargs)

    fig.update_layout(legend_title="Dataset", yaxis={"title":"Claim Value USD"}, bargap=.35)                      
    fig.update_traces(hovertemplate="Claim Amount %{y}<br> Statistic: %{x}<br>")
    fig.update_layout(yaxis=dict(tickformat='$,.2f'))

    return fig