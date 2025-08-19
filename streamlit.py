

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



st.set_page_config(layout="wide") # Use wide layout for better graph display

st.title("Railway Sales and Journey Analysis Dashboard")

# Load the cleaned data
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Ensure date columns are datetime objects, as they were saved as such
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'])
    df['Date of Journey'] = pd.to_datetime(df['Date of Journey'])

    # Ensure time columns are time objects, assuming they were saved in a compatible format
    # Use errors='coerce' and handle NaT values if conversion fails
    for col in ['Time of Purchase', 'Departure Time', 'Arrival Time', 'Actual Arrival Time']:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time

    # Handle missing 'Actual Arrival Time' after potential coercion errors if not already done
    df['Actual Arrival Time'] = df['Actual Arrival Time'].fillna(df['Arrival Time'])

    # Handle 'Railcard' column - ensure 'None' is treated as a value, not missing
    # If 'Railcard' was saved with 'None', it should load correctly as a string/object.
    # If it's still showing as missing, we might need to explicitly fillna with 'None' again just in case.
    df['Railcard'] = df['Railcard'].fillna('None')


    # Ensure 'Departure Hour' is numeric (extract again from time if needed or assume it's in CSV)
    # Assuming 'Departure Hour' is already in the CSV from previous steps,
    # but ensuring its dtype
    if 'Departure Hour' in df.columns:
        df['Departure Hour'] = pd.to_numeric(df['Departure Hour'], errors='coerce')
        df.dropna(subset=['Departure Hour'], inplace=True)
        df['Departure Hour'] = df['Departure Hour'].astype(int)
    else:
        # If not in CSV, extract from 'Departure Time'
        df['Departure Hour'] = pd.to_datetime(df['Departure Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
        df.dropna(subset=['Departure Hour'], inplace=True)
        df['Departure Hour'] = df['Departure Hour'].astype(int)


    # Ensure 'Departure Day of Week' is ordered categorical (extract again if needed)
    if 'Departure Day of Week' not in df.columns:
         df['Departure Day of Week'] = df['Date of Journey'].dt.day_name()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Departure Day of Week'] = pd.Categorical(df['Departure Day of Week'], categories=days_order, ordered=True)


    # Create 'Route' column if it doesn't exist
    if 'Route' not in df.columns:
        df['Route'] = df['Departure Station'] + ' - ' + df['Arrival Destination']


    return df

df = load_data('cleaned_railway_data.csv')

# --- Display KPIs ---
st.header("Key Revenue Performance Indicators")

total_revenue = df['Price'].sum()
total_tickets = df.shape[0]
average_price = df['Price'].mean()
median_price = df['Price'].median()
# Additional KPI: Average Price per Ticket Class (e.g., Standard vs First Class)
avg_price_by_class = df.groupby('Ticket Class')['Price'].mean().reset_index()
# Format the average price for display
avg_price_standard = avg_price_by_class[avg_price_by_class['Ticket Class'] == 'Standard']['Price'].iloc[0] if 'Standard' in avg_price_by_class['Ticket Class'].values else 0
avg_price_first_class = avg_price_by_class[avg_price_by_class['Ticket Class'] == 'First Class']['Price'].iloc[0] if 'First Class' in avg_price_by_class['Ticket Class'].values else 0


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Revenue", f"£{total_revenue:,.2f}")
with col2:
    st.metric("Total Tickets Sold", f"{total_tickets:,}")
with col3:
    st.metric("Average Ticket Price", f"£{average_price:,.2f}")
with col4:
    st.metric("Median Ticket Price", f"£{median_price:,.2f}")
with col5:
    st.metric("Avg Price (Standard Class)", f"£{avg_price_standard:,.2f}")



# --- Ticket Sales and Revenue ---
st.header("Ticket Sales and Revenue Analysis")

col_tsr1, col_tsr2 = st.columns(2)

with col_tsr1:
    st.subheader("Revenue and Ticket Count by Ticket Type")
    revenue_and_count_by_ticket_type = df.groupby('Ticket Type')['Price'].agg(['sum', 'count', 'mean']).sort_values(by='sum', ascending=False)
    revenue_and_count_by_ticket_type.rename(columns={'sum': 'Total Revenue', 'count': 'Ticket Count', 'mean': 'Average Price'}, inplace=True)
    st.dataframe(revenue_and_count_by_ticket_type.round(2))

with col_tsr2:
    st.subheader("Revenue and Ticket Count by Ticket Class")
    revenue_and_count_by_ticket_class = df.groupby('Ticket Class')['Price'].agg(['sum', 'count', 'mean']).sort_values(by='sum', ascending=False)
    revenue_and_count_by_ticket_class.rename(columns={'sum': 'Total Revenue', 'count': 'Ticket Count', 'mean': 'Average Price'}, inplace=True)
    st.dataframe(revenue_and_count_by_ticket_class.round(2))


st.subheader("Top 10 Routes Analysis")
col_routes1, col_routes2 = st.columns(2)

with col_routes1:
    st.subheader("Top 10 Routes by Ticket Count")
    route_counts = df['Route'].value_counts()
    most_popular_routes = route_counts.sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    most_popular_routes.head(10).plot(kind='bar', ax=ax3)
    ax3.set_title('Top 10 Most Popular Routes')
    ax3.set_xlabel('Route')
    ax3.set_ylabel('Number of Tickets Sold')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)

with col_routes2:
    st.subheader("Top 10 Routes by Total Revenue")
    revenue_and_count_by_route = df.groupby('Route')['Price'].agg(['sum', 'count', 'mean']).sort_values(by='sum', ascending=False)
    revenue_and_count_by_route.rename(columns={'sum': 'Total Revenue', 'count': 'Ticket Count', 'mean': 'Average Price'}, inplace=True)
    fig_route_revenue, ax_route_revenue = plt.subplots(figsize=(10, 6))
    revenue_and_count_by_route.head(10)['Total Revenue'].plot(kind='bar', ax=ax_route_revenue)
    ax_route_revenue.set_title('Top 10 Routes by Total Revenue')
    ax_route_revenue.set_xlabel('Route')
    ax_route_revenue.set_ylabel('Total Revenue (£)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_route_revenue)


# --- Peak Travel Times ---
st.header("Peak Travel Times Analysis")

col_ptt1, col_ptt2 = st.columns(2)

with col_ptt1:
    st.subheader("Distribution of Journeys by Departure Hour")
    peak_hours = df['Departure Hour'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    peak_hours.plot(kind='bar', ax=ax1)
    ax1.set_title('Distribution of Journeys by Departure Hour')
    ax1.set_xlabel('Departure Hour')
    ax1.set_ylabel('Number of Journeys')
    plt.xticks(rotation=0)
    st.pyplot(fig1)

with col_ptt2:
    st.subheader("Distribution of Journeys by Day of the Week")
    peak_days = df['Departure Day of Week'].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    peak_days.plot(kind='bar', ax=ax2)
    ax2.set_title('Distribution of Journeys by Day of the Week')
    ax2.set_xlabel('Day of the Week')
    ax2.set_ylabel('Number of Journeys')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)


# --- On-Time Performance and Delays ---
st.header("On-Time Performance and Delays")

col_otp1, col_otp2 = st.columns(2)

with col_otp1:
    st.subheader("Distribution of Journey Status")
    journey_status_counts = df['Journey Status'].value_counts()
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    journey_status_counts.plot(kind='bar', ax=ax4)
    ax4.set_title('Distribution of Journey Status')
    ax4.set_xlabel('Journey Status')
    ax4.set_ylabel('Number of Journeys')
    plt.xticks(rotation=0)
    st.pyplot(fig4)

with col_otp2:
    st.subheader("Top Reasons for Delay")
    delayed_df = df[df['Journey Status'] == 'Delayed'].copy()
    reason_for_delay_counts = delayed_df['Reason for Delay'].value_counts().sort_values(ascending=False)
    reason_for_delay_counts = reason_for_delay_counts[reason_for_delay_counts.index != 'no delay'] # Exclude 'no delay'
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    reason_for_delay_counts.head(10).plot(kind='bar', ax=ax5)
    ax5.set_title('Top Reasons for Delay')
    ax5.set_xlabel('Reason for Delay')
    ax5.set_ylabel('Number of Delays')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig5)

st.subheader("Delay Duration for Delayed Journeys (Minutes)")
delayed_df = df[df['Journey Status'] == 'Delayed'].copy()
# Ensure time columns are strings before combining with date
delayed_df['Scheduled Arrival Datetime'] = pd.to_datetime(delayed_df['Date of Journey'].astype(str) + ' ' + delayed_df['Arrival Time'].astype(str), errors='coerce')
delayed_df['Actual Arrival Datetime'] = pd.to_datetime(delayed_df['Date of Journey'].astype(str) + ' ' + delayed_df['Actual Arrival Time'].astype(str), errors='coerce')

# Drop rows with NaT values resulting from coercion errors
delayed_df.dropna(subset=['Scheduled Arrival Datetime', 'Actual Arrival Datetime'], inplace=True)

# Handle cases where the actual arrival is on the next day
delayed_df.loc[delayed_df['Actual Arrival Datetime'] < delayed_df['Scheduled Arrival Datetime'], 'Actual Arrival Datetime'] += pd.Timedelta(days=1)

delayed_df['Delay Duration'] = delayed_df['Actual Arrival Datetime'] - delayed_df['Scheduled Arrival Datetime']
delayed_df['Delay Duration Minutes'] = delayed_df['Delay Duration'].dt.total_seconds() / 60

# Create a box plot for delay duration
fig_boxplot, ax_boxplot = plt.subplots(figsize=(9, 5))
sns.boxplot(x=delayed_df['Delay Duration Minutes'], ax=ax_boxplot)
ax_boxplot.set_title('Distribution of Delay Duration (Minutes)')
ax_boxplot.set_xlabel('Delay Duration Minutes')
st.pyplot(fig_boxplot)


# --- Cancellation and Refund Analysis ---
st.header("Cancellation and Refund Analysis")

col_cr1, col_cr2 = st.columns(2)

with col_cr1:
    st.subheader("Top Reasons for Cancellation")
    cancelled_df = df[df['Journey Status'] == 'Cancelled'].copy()
    cancellation_reasons = cancelled_df['Reason for Delay'].value_counts().sort_values(ascending=False)
    cancellation_reasons = cancellation_reasons[cancellation_reasons.index != 'no delay'] # Exclude 'no delay'
    fig_cancellation, ax_cancellation = plt.subplots(figsize=(10, 6))
    cancellation_reasons.head(10).plot(kind='bar', ax=ax_cancellation)
    ax_cancellation.set_title('Top Reasons for Cancellation')
    ax_cancellation.set_xlabel('Reason for Delay')
    ax_cancellation.set_ylabel('Number of Cancellations')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_cancellation)

with col_cr2:
    st.subheader("Reasons for Refund Requests")
    refund_requested_df = df[df['Refund Request'] == 'Yes'].copy()
    refund_reasons_from_delay = refund_requested_df['Reason for Delay'].value_counts().sort_values(ascending=False)
    refund_reasons_from_delay = refund_reasons_from_delay[refund_reasons_from_delay.index != 'no delay'] # Exclude 'no delay'
    fig_refund, ax_refund = plt.subplots(figsize=(10, 6))
    refund_reasons_from_delay.head(10).plot(kind='bar', ax=ax_refund)
    ax_refund.set_title('Reasons for Refund Requests (based on Reason for Delay)')
    ax_refund.set_xlabel('Reason for Delay')
    ax_refund.set_ylabel('Number of Refund Requests')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_refund)


st.write(f"Total number of refund requests: {len(refund_requested_df)}")


# --- Additional Visualizations ---
st.header("Additional Insights")

col_add_viz1, col_add_viz2 = st.columns(2)

with col_add_viz1:
    st.subheader("Price Distribution (Histogram)")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, bins=30, ax=ax_hist)
    ax_hist.set_title('Distribution of Price')
    ax_hist.set_xlabel('Price')
    ax_hist.set_ylabel('Frequency')
    st.pyplot(fig_hist)

with col_add_viz2:
    st.subheader("Ticket Count by Railcard Usage")
    railcard_counts = df['Railcard'].value_counts()
    fig_railcard, ax_railcard = plt.subplots(figsize=(10, 6))
    railcard_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax_railcard, startangle=90)
    ax_railcard.set_title('Distribution of Ticket Sales by Railcard Usage')
    ax_railcard.set_ylabel('') # Hide the default y-label for pie charts
    st.pyplot(fig_railcard)