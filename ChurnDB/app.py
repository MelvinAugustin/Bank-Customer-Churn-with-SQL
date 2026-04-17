import streamlit as st
import pandas as pd

df = pd.read_csv("Churn.csv")

st.title("Customer Churn Dashboard")

#clean churn
df["churn_flag"] = df["churn"]


# KPIs
total_customers = len(df)
churned_customers = df["churn_flag"].sum()
churn_rate = round((churned_customers / total_customers) * 100, 2)

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", int(churned_customers))
col3.metric("Churn Rate (%)", churn_rate)

st.divider()


#data display
st.subheader("Customer Data")
st.dataframe(df)

# at risk customers
st.subheader("At Risk Customers")

risk_df = df[
    (df["balance"] < 1000) |
    (df["active_member"] == 0)
]

st.dataframe(risk_df)


#charts
st.subheader("Churn Distribution")
st.bar_chart(df["churn"].value_counts())

st.subheader("Churn vs Balance")
st.scatter_chart(df[["balance", "churn"]])