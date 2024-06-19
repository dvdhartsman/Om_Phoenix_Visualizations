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
    # if "claim_amount" in data.columns:
    #     data = data.rename(columns={"claim_amount":"total_claim_amount"})

    male_data = data.query("gender == 'Male'")['claim_amount']
    female_data = data.query("gender == 'Female'")['claim_amount']

    male_median_x = male_data.median().round(2)
    female_median_x = female_data.median().round(2)  

    # KDEs fig.update_layout(yaxis=dict(tickformat='$,.2f'))
    male_kde = ff.create_distplot([male_data], group_labels=['Male'], show_hist=False, show_rug=False)
    
    female_kde = ff.create_distplot([female_data], group_labels=['Female'], show_hist=False, show_rug=False)

    # Create subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=(f'Men - Median Claim ${male_median_x:,.2f}', f'Women - Median Claim ${female_median_x:,.2f}', 'Male vs Women Overlaid'),
                        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"colspan": 2}, None]],
                        row_heights=[0.5, 0.5],
                        column_widths=[0.5, 0.5])
    

    # Male KDE Plot
    for trace in male_kde['data']:
        trace["hoverinfo"] = 'x'
        trace["xhoverformat"] = "$,.2f"
        trace["showlegend"] = False
        trace["hovertemplate"] = 'Claim Amount: %{x:$,.2f}'
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
        trace["xhoverformat"] = "$,.2f"
        trace["showlegend"] = False
        trace["hovertemplate"] = 'Claim Amount: %{x:$,.2f}'
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
                             hoverinfo='x', xhoverformat="$,.2f", hovertemplate = 'Claim Amount: %{x:$,.2f}'), 
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=female_kde['data'][0]['x'], y=female_kde['data'][0]['y'], 
                             mode='lines', name='Female', fill='tozeroy', line=dict(color='lightcoral'), opacity=0.1,
                             hoverinfo='x', xhoverformat="$,.2f", hovertemplate = 'Claim Amount: %{x:$,.2f}'), 
                  row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1200, title_text="Claim Distribution - Men vs Women: Higher Peaks Indicate More-Common Claim Amounts")
    fig.update_xaxes(title_text="Total Claim in USD", row=1, col=1)
    fig.update_xaxes(title_text="Total Claim in USD", row=1, col=2)
    fig.update_xaxes(title_text="Total Claim in USD", row=2, col=1)

    fig.update_layout(showlegend=True, legend=dict(x=0.875, y=0.275))
    fig.update_yaxes(showticklabels=False)

    # Show plot
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
    fig = px.histogram(data_frame=data["age"], labels={"age":"Age"}, title="Distribution of Claims by Age", **kwargs)
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
    fig.update_traces(hovertemplate="Claim Amount: %{x} <br>Group: %{y}")
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
                       title=f"Claim Distribution by Value - {condition}", nbins=20, **kwargs)
    fig.update_layout(legend_title="", xaxis={"title":"Claim Value"}, yaxis={"title":"Number of Claims"})
    fig.update_traces(name="Claims", marker_line_color='black', marker_line_width=1.5)
    fig.update_layout(xaxis=dict(tickformat='$,.2f'))
    
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
    
# ---------------------------------------- main function ------------------------------------------------------------------

def main():

    # Medical Practice Data -------------------------------------------------------------------------------
    df_med = pd.read_csv("Kaggle_medical_practice_20.csv", index_col=0)

    # Medical Data Basic Processing ----------------------------------------------------
    # Attorney
    df_med["private_attorney"] = df_med["private_attorney"].map({0:"No", 1:"Yes"})
    
    # Claim amount
    df_med.rename(columns={"total_claim_amount":"claim_amount"}, inplace=True)
    
    # Marital Status
    df_med["marital_status"] = df_med["marital_status"].map({0:"Divorced", 1:"Single", 2:"Married", 3:"Widowed",4:"Unknown"})
    
    # Age Bins # 1 
    # bins = [-np.inf, 2, 12, 18, 35, 60, np.inf]
    # labels = ["Infant 0-2", "Child 2-12", "Teenager 12-18", "Young Adult 18-35",
    #       "Adult 35-60", "Senior Citizen 60+"]
    
    # Age Bins # 2
    bins = [ -np.inf,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
       85, np.inf]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30","30-35", "35-40", "40-45", "45-50", "50-55","55-60",
              "60-65", "65-70","70-75", "75-80", "80-85", "85+"]
    
    df_med["age_bracket"] = pd.cut(df_med["age"], bins=bins, labels=labels)

    # Severity ints to string categories
    severity_mapping = {
            1: 'Emotional Trauma', 2: 'Low', 3: 'Moderate', 4: 'Moderately High',
            5: 'High', 6: 'Very High', 7: 'Severe', 8: 'Very Severe', 9: 'Death'
        }
        # Create severity category column
    df_med['severity_category'] = df_med['severity'].map(severity_mapping)


    # Car Insurance Data -----  ----------  -----------  --------------  -------------  ---------------  -------------------
    df_ins = pd.read_csv("Insurance_claims_mendeleydata_6.csv")

    # Process the insurance Data
    df_ins = preprocess_insurance_data(df_ins)

    st.header("**Analysis:**")
    graph_description = """
As you read through the analysis, we would also like you to be aware that these visualizations are interactive. 
Most of the plots will allow you to zoom in on a region by clicking and dragging over an area with your mouse. 
Also, if there is a legend in the upper-right of a visualization, you can click on an item in the legend to toggle that group on/off.
"""
    st.markdown(graph_description)
    st.subheader("Choose one of our 2 datasets for analysis")
    data_source = st.selectbox("Choose Data", ["Medical Malpractice", "Auto Insurance Claims"])

    
    # -------------First ---------------------------------------Section------------------------------------------------
    # ------------------------- MEDICAL PRACTICE ------------------------------------------------------------------
    if data_source == "Medical Malpractice": 
        data = df_med
        st.subheader("You have selected the Medical Malpractice dataset")
        st.markdown("This dataset contains records of claims against medical practitioners from 20 different distinct specializations.")
        st.markdown("---")
        st.subheader("1. Gender:")
        medical_gender_analysis = """
Male and Female malpractice claims exhibit some interesting differences in their distributions. 
Women receive \$44,873 more than Men in median claim amounts, but men *average* about \$16,500 more per claim than women.
Women generally receive larger payouts for claims across different insurance types, with private insurance being the exception. 
Men receive \$14,378 more than women for median claims for claims paid by private insurance.
Also, in spite of men representing less than 40\% of claims in our data, they generally enjoy a 50\%-or-larger share of claims above \$300k. 
This may be due to men being primary earners for their households, subsequently receiving more for anticipated lost wages.
"""
        st.markdown(medical_gender_analysis)
        st.plotly_chart(plotly_pie(data, "gender", color_discrete_sequence=["lightcoral","blue"]))
        st.plotly_chart(plotly_gender(data))
        
        # st.subheader("Gender and Insurance Types")
        # st.plotly_chart(plotly_box_gender(data))
        ins_types_gender_stats = df_med.groupby(["gender", "insurance"])["claim_amount"]\
                               .agg("mean").round(2).reset_index()
        overall_gender_stats = df_med.groupby("gender")["claim_amount"].mean().round(2).reset_index()

        gender_averages = pd.concat([ins_types_gender_stats, (overall_gender_stats)]).fillna("All Data")
        gen_fig = px.bar(data_frame= gender_averages, y="insurance", x="claim_amount", color="gender", 
                               barmode="group", title="Overall Average Male and Female Claim Value and Averages for Different Insurance Types",
                               color_discrete_sequence=["lightcoral","blue"], width=1200, height=600,
                               labels={"gender":"Gender", "insurance":"Insurance", "claim_amount":"Claim Amount"})\
                                .update_layout(xaxis=dict(tickformat='$,.2f', title="Claim Amount"), yaxis=dict(title="Insurance"),
                                               legend_title="Gender")
        
        st.plotly_chart(gen_fig)
        
        
        # Bins #1
        bins = [-np.inf, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000,
                                       550_000, 600_000, 650_000, 700_000, 750_000, 800_000, 850_000, 900_000, np.inf]

        labels = ["<$200k", "$200-250k", "$250-300k", "$300-350k", "$350-400k", "$400-450k", "$450-500k", "$500-550k", "$550-600k",
                "$600-650k", "$650-700k", "$700-750k", "$750-800k", "$800-850k", "$850-900k", ">$900k"]

        len(labels)

        # Create New Feature for Claim Brackets
        data["claim_category"] = pd.cut(data["claim_amount"], bins=bins, labels=labels)
        
        
        st.plotly_chart(px.line(data_frame = data.groupby("claim_category")["gender"].value_counts(normalize=True).multiply(100)\
                                .round(2).reset_index(),x="claim_category", y="proportion", color="gender", 
                                color_discrete_sequence=["lightcoral","blue"], labels={"gender":"Gender", 
                                                                                       "proportion":"Percentage",
                                                                                       "claim_category":"Value Range"},
                                title="Men Represent a Larger Proportion of Large-Amount Claims Than Women", markers=True)\
                                .update_layout(yaxis=dict(tickprefix='%')))

        # Age ---------------------------------
        st.subheader("2. Age:")
        age_analysis = """
Highest Median Claim Amount: The median total claim amount is highest for infants at \$164,752, 
indicating costly medical care likely due to intensive care needs.
Children (2-12 years) have a median claim amount of \$116,237, slightly higher than teenagers (\$115,797), 
suggesting significant medical costs for these age groups.
Adults have a median claim amount of \$14,644 higher than young adults, indicating increased medical costs with age, 
possibly due to consistent medical management, lower incidence of high-cost accidents, or different insurance coverage.
Senior Citizens (60+ ) have a median claim amount of \$70,213, the lowest among all age groups, 
possibly due to consistent medical management, lower accident incidence, or tailored insurance policies.

"""

        st.markdown(age_analysis)

        # Line Plot of Median Claims by Age
        # st.plotly_chart(plotly_age(data))

        st.plotly_chart(plotly_age_hist(data, color_discrete_sequence=["sienna"]))

        # Mean and Median Age Barplots
        st.plotly_chart(plotly_age_bracket(data, template="seaborn"))

        st.plotly_chart(plotly_age_line(data, "age_bracket", template="seaborn"))

        # ------------------------ MARIAM's CODE ----------------   ATTORNEYS  ------------------------
        st.subheader("3. Attorney Involvement:")
        attorney_analysis = """
Hiring a private attorney correlates with a substantially higher median total claim amount of \$113,823, 
compared to \$46,444 for cases where no private attorney was involved.
This indicates a potential 145\% increase in median claim amounts when a private attorney is engaged.
"""
        st.markdown(attorney_analysis)
        st.plotly_chart(plotly_pie(data, "private_attorney", template="presentation"))
        # Attorney
        st.plotly_chart(plotly_mean_median_bar(data,"private_attorney", color_discrete_sequence=["mistyrose", "sienna"]))

        # MARITAL STATUS --------------------------
        st.subheader("4. Marital Status:")  # --------------- MARITAL STATUS
        marital_analysis = """
The median total claim amount is highest for divorced individuals at \$339,190, 
indicating potentially higher settlements related to divorce proceedings.
Single individuals exhibit a moderate median claim amount of \$102,312, 
reflecting diverse financial circumstances and varying insurance coverage, 
which is approximately 2\% higher than that of married individuals.
Those with unknown marital status have the lowest median claim amount at \$47,931, 
suggesting incomplete or undisclosed marital status data impacting claim outcomes.
"""
        st.markdown(marital_analysis)
        st.plotly_chart(plotly_pie(data, "marital_status", template="presentation"))
        # Marital Status
        st.plotly_chart(plotly_mean_median_bar(data, "marital_status", template="plotly_dark"))

        # Severity
        # st.plotly_chart(plotly_mean_median_bar(data, "severity"))

        # ------------------------ Indrajith's Code --------------------------------------------------

        # Severity Analysis
        # Mapping severity levels to categories
        st.subheader("5. Severity:")
        severity_points = """
                    - We will fill this later, i need inputs from dataset team <br>
                    """
        st.markdown(
            f"""
                <style>
                .text-area-wrapper {{
                    background-color: rgba(128, 128, 128, 0.3); /* Grey with 30% opacity */
                    padding: 10px;
                    border-radius: 5px;
                }}
                </style>
                <div class="text-area-wrapper" ">{severity_points}</div>
                """,
            unsafe_allow_html=True
        )

        # Define the order of severity categories
        severity_order = ['Emotional Trauma', 'Low', 'Moderate', 'Moderately High',
                          'High', 'Very High', 'Severe', 'Very Severe', 'Death']

        # Figure Creation
        severity_avg_claim = data.groupby('severity_category')[
            'claim_amount'].mean().round(2).reset_index()

        fig_severity_claim = px.bar(severity_avg_claim, x='severity_category', y='claim_amount',
                                    labels={'x': 'Severity Category', 'y': 'Average Claim Amount',
                                             "claim_amount":"Claim Amount", "severity_category":"Severity"},
                                    title='Average Claim Amount by Severity Category',
                                    color='claim_amount', color_continuous_scale='blues')


        fig_severity_claim.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': severity_order},
                                         xaxis_title="Severity Category", yaxis_title="Average Claim amount in USD")
        
        fig_severity_claim.update_layout(legend_title="Claim Amount")
        
        # Traces for template
        fig_severity_claim.update_traces(hovertemplate="Claim Amount %{y}<br> Severity: %{x}<br>")
        
        # Layout for format
        fig_severity_claim.update_layout(yaxis=dict(tickformat='$,.2f'))
        # show the plot
        # fig_severity_claim.update_traces(dict(hovertemplate={'y'}, yhoverformat="$,.2f"))
        st.plotly_chart(fig_severity_claim)

        #### MEDICAL SPECIALTY
        st.subheader("6. Medical Specialty:")
        specialty_analysis = """
Pediatrics, Dermatology, and Urological Surgery exhibit the highest median total claim amounts, 
indicating potentially higher costs and complexities in medical treatments within these specialities.
Radiology, Anesthesiology, and Emergency Medicine have the lowest median total claim amounts, 
reflecting comparatively lower costs associated with their respective medical practices.
"""
        st.markdown(specialty_analysis)
        # Medical Specialty ### VERY IMPORTANT!!!!!!!!! ##############
        st.plotly_chart(plotly_injury_bar(data, "specialty", template="seaborn"))
# .update_layout(yaxis=dict(tickformat='$,.2f')

        # ---------------------------------- FILTERS -----------------------------------------------
        st.header("Try Out Multiple Filters:")
        st.write('If you would like to deactivate a filter select: "None"')
        
        # Age -------
        min_age, max_age = st.slider("Age Range", min_value = data["age"].min().astype(int), max_value=data["age"].max().astype(int), \
                        value=(data["age"].min().astype(int), data["age"].max().astype(int)), step=1)
        
        # Boolean Mask for the Filter
        age_condition = (data["age"] >= min_age) & (data["age"] <= max_age)

        # SEVERITY -----------
        severity_type_status = st.selectbox("Severity of Accident:", [None] + list(data["severity"].sort_values().unique()),index=0)
        if severity_type_status:
            severity_type_condition = (data["severity"] == severity_type_status)
        else:
            severity_type_condition = True
        
        # private_attorney -----------
        attorney_type_status = st.selectbox("Private Attorney Involved:", [None] + list(data["private_attorney"].unique()),index=0)
        if attorney_type_status:
            attorney_type_condition = (data["private_attorney"] == attorney_type_status)
        else:
            attorney_type_condition = True

        # marital_status -----------
        marital_status_type_status = st.selectbox("Marital Status:", [None] + list(data["marital_status"].unique()),index=0)
        if marital_status_type_status:
            marital_status_type_condition = (data["marital_status"] == marital_status_type_status)
        else:
            marital_status_type_condition = True

        # specialty -----------
        specialty_type_status = st.selectbox("Medical Specialty:", [None] + list(data["specialty"].unique()),index=0)
        if specialty_type_status:
            specialty_type_condition = (data["specialty"] == specialty_type_status)
        else:
            specialty_type_condition = True

        # insurance -----------
        insurance_type_status = st.selectbox("Type of Insurance:", [None] + list(data["insurance"].unique()),index=0)
        if insurance_type_status:
            insurance_type_condition = (data["insurance"] == insurance_type_status)
        else:
            insurance_type_condition = True

        # gender -----------
        gender_type_status = st.selectbox("Gender:", [None] + list(data["gender"].unique()),index=0)
        if gender_type_status:
            gender_type_condition = (data["gender"] == gender_type_status)
        else:
            gender_type_condition = True
        

        ### COLLECTING CONDITIONS  -----------------------------------------------------
        all_conditions = age_condition & severity_type_condition & attorney_type_condition & marital_status_type_condition\
        & specialty_type_condition & insurance_type_condition & gender_type_condition

        st.markdown("---")
        st.write("Here's a brief summary of claims for the data you have selected:")
        
        ### SUMMARY PLOTS
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
        st.plotly_chart(plotly_filtered_claims_bar(description_table, template="plotly_dark"))

        # Scatterplot of Claim vs Age
        keys = [None] + list(data.select_dtypes(exclude=np.number).columns.str.title().sort_values().str.replace("_", " "))
        values = [None] + sorted(list(data.select_dtypes(exclude=np.number).columns), key=lambda x: x.lower())
        age_col_dict = dict(zip(keys, values))

        st.subheader("Use the Scatterplot to Explore Groups from the Data")
        group = st.selectbox("Add Detail for Subsets:", keys)
        st.plotly_chart(plotly_scatter_age(data[all_conditions], age_col_dict[group]))


    # -------------------------------------------------------------------------------------------------
    # ------------------------ INSURANCE DATA ---------------------------------------------------------
    elif data_source == "Auto Insurance Claims":
        data = df_ins
        st.subheader("This dataset is comprised of car accident claims.")
        st.write("These claims were all recorded between January 1, 2015 and March 1, 2015")
        st.markdown("---")
        
        # Gender
        st.subheader("1. Gender:")
        
        gender_paragraph = """"Gender" refers to the policy holder (liable party) for this dataset. 
        The analysis of total claim amounts by gender reveals that 
        female policyholders tend to pay slightly larger claims compared to male policyholders.
The mean (average) claim amount for females is \$51,169, 
which is 3.94\% higher than the male mean of \$49,230.
The median, representing the middle value of all claims, 
is higher for females at \$57,120 compared to \$55,750 for males, showing a 2.46\% difference.
The mean provides an overall average but can be affected by extremely high or low claims.
The median offers a better sense of the typical claim amount.
Both metrics indicate that, on average, female policyholders report similar but mariginally higher claim amounts than their male counterparts.
"""
        
        st.markdown(gender_paragraph)

        st.plotly_chart(plotly_gender(data))
        
        # age_bracket
        st.subheader("2. Age:")

        age_paragraph = """
The ages in this dataset are confined to a fairly narrow range. 80\% of all claimants fall between
28-51 years old. As age increases, so does the average claim amount, with victims aged 60-65 having 
the highest average claims at \$58,900. Overall, there is a discernible upward trend in claim amounts with increasing age, 
underscoring a positive correlation between age and claim size. This likely indicates that older 
drivers tend to own more expensive cars, leading to higher claim settlements.
"""
# These findings highlight the significance of age as a determinant in assessing 
# insurance claim risks and guiding strategic policy adjustments.

        st.markdown(age_paragraph)
        st.plotly_chart(plotly_age_hist(data, color_discrete_sequence=["sienna"]))
        st.plotly_chart(plotly_age_bracket(data, template="seaborn"))
        st.plotly_chart(plotly_age_line(data, "age_bracket", template="seaborn"))

        # Make of Car -> probably not that important
        st.subheader("3. Auto Manufacturer:")
        auto_paragraph = """
Nissan, Saab, and Subaru comprise the largest proportions of manufacturers in our data with 9\% each respectively.
Claims involving Ford vehicles have the highest median claim amount of around \$63,500 followed closely by
luxury brand BMW at \$62,480. The largest and smallest average claims for each manufacturer are separated by a range of 
around \$11,000 from the highest average, BMW at \$54,000 to the lowest average, Toyota, at \$43,000.

"""
        st.markdown(auto_paragraph)

        # Treemap
        fig = px.treemap(data[["auto_make", "auto_model"]].value_counts(normalize=True).round(2).reset_index(), 
                         path=["auto_make", "auto_model"], values="proportion", title="Distribution of Makes and Models")
        
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        fig.update_traces(hovertemplate="Vehicle %{label}<br>Percentage of Records %{value:.1%}")
        st.plotly_chart(fig)
        
        st.plotly_chart(plotly_injury_bar(data, "auto_make"))
        
        st.plotly_chart(plotly_injury_bar(data,"auto_model"))
        

        # auto_year -> CURIOUS DATA, implies older cars are of a higher claim value
        st.subheader("4. Model year:")
        model_year_paragraph = """The analysis of auto year and claim amounts for this dataset indicate
          older model years tend to receive higher average and median claim amounts compared to newer models.
For example, vehicles from the mid-to-late 1990s and early 2000s (e.g., 1995-2005) show higher average claim amounts, 
ranging from \$47,134 to \$57,535. This suggests potentially higher repair costs or difficult-to-locate parts,
and perhaps more adverse outcomes due to deteriorating safety features within the vehicle.
In contrast, newer model years from 2010 onwards (e.g., 2010-2015) demonstrate lower average claim amounts,
ranging from \$42,853 to \$48,323, likely indicating improved vehicle safety and durability standards.

"""
        st.markdown(model_year_paragraph)
        st.plotly_chart(plotly_mean_median_bar(data,"auto_year", template="presentation")\
                        .update_layout(xaxis=dict(tickvals=np.arange(1995, 2016))))
        st.plotly_chart(plotly_age_line(data, "auto_year", template="presentation"))
        

        # States
        st.subheader("5. State:")

        state_paragraph ="""
        New York (NY) makes up the largest proportion of claims in our data, and it also has the largest
          mean and median claim values at roughly \$54,250 and \$58,675 respectively. With the exception of 
          Ohio, all the states represented in our data are on the east coast. Ohio has the smallest average and median claims
          of all the states present while also representing the fewest claims in the data.
"""
        st.markdown(state_paragraph)
        st.plotly_chart(plotly_pie(data, "state", template="presentation"))
        st.plotly_chart(plotly_states(data))
        # st.plotly_chart(plotly_box_states(data)) # Removed to avoid over complication

        # Incident Date showed a relatively stationary time series, not a lot of inferential value
        
        # accident_type
        st.subheader("6. Accident Type:")
        
        accident_paragraph ="""
        Claims involving moving vehicles (single or multi-vehicle) have similarly sized claim values, 
        whereas claims for unattended vehicles typically have much smaller claim values. This disparity can be explained by the absence of 
        physical injuries for claims involving vehicle theft or damage to a parked car.  
        Multi-vehicle and single-vehicle collisions average roughly \$62,000 and \$63,500 respectively per claim.
Parked car and vehicle theft incidents have notably lower average claim amounts at \$5,300 and \$5,500.

"""
        st.markdown(accident_paragraph)

        st.plotly_chart(plotly_pie(data, "accident_type", template="presentation"))
        st.plotly_chart(plotly_mean_median_bar(data, "accident_type", template="seaborn"))

        # collision_type
        st.subheader("7. Collision Type:")

        collision_paragraph ="""Front collisions account for 24.4\% of all claims, with an average payout of \$64,777 and 
        a median of \$63,950. This makes logical sense, as front-end collisions will endanger the driver, any other front-seat
        occupants, and the components of the engine. Side and rear collisions receive incrementally smaller claim sizes, and as 
previously described, unattended vehicle claims receive substantially less.
"""
        st.markdown(collision_paragraph)

        st.plotly_chart(plotly_pie(data, "collision_type", template="presentation"))
        st.plotly_chart(plotly_mean_median_bar(data, "collision_type"))

        # incident_severity
        st.subheader("8. Incident Severity:")

        severity_paragraph ="""The two most frequent incident severities found in this data are minor damage, representing 42\% of all claims,
        and total losses representing 32.4\% of all claims. Major Damage incidents have the highest average claim amount at almost \$64,000,
Total losses follow closely behind with an average claim value of \$61,792. Total losses having lower claim amounts than major damage
is most likely attributed to insurers wanting to save costs on mechanical labor and parts. Often, insurance companies will prefer to 
"total out" a car rather than repair it if the estimated repair costs are larger than the appraised value of a car. 
Minor Damage claims average \$48,600, well below the previous 2 categories. Trivial Damage claims receive significantly 
smaller compensation at \$5,300 on average. 
"""
        st.markdown(severity_paragraph)
        st.plotly_chart(plotly_pie(data, "incident_severity", template="presentation"))
        st.plotly_chart(plotly_mean_median_bar(data, "incident_severity", template="seaborn"))

        # bodily_injuries
        st.subheader("9. Number of Bodily Injuries:")

        bodily_injuries_paragraph ="""This data has a balanced proportion of claims with 0, 1, and 2 bodily injuries.
        Claims without bodily injuries have a median value of \$56,700, 
        while those with one injury surprisingly have a slightly lower median value of \$54,000. Claims involving 
        two injuries have the largest median value of \$57,935 as expected. 
"""
        st.markdown(bodily_injuries_paragraph)

        
        st.plotly_chart(plotly_pie(data, "bodily_injuries", template="presentation"))
        st.plotly_chart(plotly_mean_median_bar(data, "bodily_injuries", color_discrete_sequence=["chocolate", "gray"]))

        # authorities_contacted
        st.subheader("10. Authorities Contacted:")

        authorities_paragraph ="""There are 4 distinct categories of authorities listed in our data:
        Fire, Ambulance, Police, and "Other". For the specifically named authorities, cases involving the fire department 
        received the largest claims. They have an average claim amount of about \$61,439 and a median value of \$60,000. 
        Next largest, incidents where an ambulance was contacted have an average claim amount of approximately 
        \$60,357, with a median value of \$59,300. Third, incidents requiring police intervention have a lower mean claim amount of roughly \$44,193, 
with a median of \$51,800.  "Other" authorities had the largest average claim amount of around \$65,156 
and a median value of \$64,080. While police do address issues that result in large claims, they seem to
handle the overwhelming majority of small claim instances. Notice how *only police* handle claims with values
less than \$18,000
"""
        st.markdown(authorities_paragraph)

        st.plotly_chart(plotly_pie(data, "authorities_contacted", template="presentation"))
        st.plotly_chart(plotly_mean_median_bar(data, "authorities_contacted", template="plotly"))
        st.plotly_chart(plotly_scatter_age(data, "authorities_contacted"))

        # police_report_available
        st.subheader("11. Police Report:")

        police_report_paragraph ="""Incidents with a police report available had an average claim amount of 
        \$52,083, which is approximately 11.5\% higher than incidents without a report (\$46,738).
The median claim amount for incidents with a police report was \$57,110, showing a difference of about 
3\% lower compared to incidents without a report (\$55,500).
 Cases where the availability of a police report was unknown showed a mean claim amount of 
 \$52,171 and a median of \$58,050.

"""
        st.markdown(police_report_paragraph)

        st.plotly_chart(plotly_pie(data, "police_report_available", template="presentation"))
        st.plotly_chart(plotly_mean_median_bar(data, "police_report_available", color_discrete_sequence=["blue", "lightgrey"]))

        # --------------------------------- Filtering Conditions -------------------------------------------

        st.header("Try Out Multiple Filters:")
        st.write('If you would like to deactivate a filter select: "None"')
        
        # Age -------
        min_age, max_age = st.slider("Age Range", min_value = data["age"].min().astype(int), max_value=data["age"].max().astype(int), \
                        value=(data["age"].min().astype(int), data["age"].max().astype(int)), step=1)
        
        # Boolean Mask for the Filter
        age_condition = (data["age"] >= min_age) & (data["age"] <= max_age)
        
        # gender -----------
        gender_type_status = st.selectbox("Gender:", [None] + list(data["gender"].unique()),index=0)
        if gender_type_status:
            gender_type_condition = (data["gender"] == gender_type_status)
        else:
            gender_type_condition = True

        # accident_type -----------
        accident_type_type_status = st.selectbox("Accident Type:", [None] + list(data["accident_type"].unique()),index=0)
        if accident_type_type_status:
            accident_type_condition = (data["accident_type"] == accident_type_type_status)
        else:
            accident_type_condition = True

        # collision_type -----------
        collision_type_status = st.selectbox("Collision Type:", [None] + list(data["collision_type"].unique()),index=0)
        if collision_type_status:
            collision_type_condition = (data["collision_type"] == collision_type_status)
        else:
            collision_type_condition = True

        # incident_severity -----------
        incident_severity_type_status = st.selectbox("Incident Severity:", [None] + list(data["incident_severity"].unique()),index=0)
        if incident_severity_type_status:
            incident_severity_type_condition = (data["incident_severity"] == incident_severity_type_status)
        else:
            incident_severity_type_condition = True

        # authorities_contacted -----------
        authorities_contacted_type_status = st.selectbox("Authorities Contacted:", [None] + \
                                                         list(data["authorities_contacted"].dropna().unique()),index=0)
        if authorities_contacted_type_status:
            authorities_contacted_condition = (data["authorities_contacted"] == authorities_contacted_type_status)
        else:
            authorities_contacted_condition = True


        # state -----------
        state_type_status = st.selectbox("State:", [None] + list(data["state"].unique()),index=0)
        if state_type_status:
            state_condition = (data["state"] == state_type_status)
        else:
            state_condition = True


        # property_damage -----------
        property_damage_type_status = st.selectbox("Property Damage:", [None] + list(data["property_damage"].unique()),index=0)
        if property_damage_type_status:
            property_damage_condition = (data["property_damage"] == property_damage_type_status)
        else:
            property_damage_condition = True


        # bodily_injuries -----------
        bodily_injuries_type_status = st.selectbox("Number of Bodily Injuries:", [None] + list(data["bodily_injuries"].unique()),index=0)
        if bodily_injuries_type_status:
            bodily_injuries_condition = (data["bodily_injuries"] == bodily_injuries_type_status)
        else:
            bodily_injuries_condition = True


        # police_report_available -----------
        police_report_available_type_status = st.selectbox("Police Report Available?:", [None] + list(data["police_report_available"].unique()),index=0)
        if police_report_available_type_status:
            police_report_available_condition = (data["police_report_available"] == police_report_available_type_status)
        else:
            police_report_available_condition = True


        # auto_make -----------
        auto_make_type_status = st.selectbox("Auto Make:", [None] + list(data["auto_make"].unique()),index=0)
        if auto_make_type_status:
            auto_make_condition = (data["auto_make"] == auto_make_type_status)
        else:
            auto_make_condition = True


        # auto_model -----------
        auto_model_type_status = st.selectbox("Auto Model:", [None] + list(data["auto_model"].unique()),index=0)
        if auto_model_type_status:
            auto_model_condition = (data["auto_model"] == auto_model_type_status)
        else:
            auto_model_condition = True


        # auto_year -----------
        auto_year_type_status = st.selectbox("Auto Year:", [None] + list(data["auto_year"].sort_values(ascending=False).unique()),index=0)
        if auto_year_type_status:
            auto_year_condition = (data["auto_year"] == auto_year_type_status)
        else:
            auto_year_condition = True
        

        ### COLLECTING CONDITIONS  -----------------------------------------------------
        all_conditions = age_condition & gender_type_condition & accident_type_condition\
        & collision_type_condition & incident_severity_type_condition & authorities_contacted_condition\
        & state_condition & property_damage_condition & bodily_injuries_condition & police_report_available_condition\
        & auto_make_condition & auto_model_condition & auto_year_condition

        st.markdown("---") 

        st.subheader("Here's a breakdown of the data you have selected:")
        
        ### SUMMARY PLOTS
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
        st.plotly_chart(plotly_filtered_claims_bar(description_table, template="plotly_white"))

        # Scatterplot of Claim vs Age
        keys = [None] + list(data.select_dtypes(exclude=np.number).columns.str.title().sort_values().str.replace("_", " "))
        values = [None] + sorted(list(data.select_dtypes(exclude=np.number).columns), key=lambda x: x.lower())
        age_col_dict = dict(zip(keys, values))


        st.subheader("Use the Scatterplot to Explore Groups from the Data")
        group = st.selectbox("Select Subsets of the Data:", keys)
        st.plotly_chart(plotly_scatter_age(data[all_conditions], age_col_dict[group]))


        # -------------------------------- Summary with Filters -----------------------------------



if __name__ == "__main__":
    main()

    
