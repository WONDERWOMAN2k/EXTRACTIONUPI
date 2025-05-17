# app.py

import pdfplumber
import pandas as pd
import re
import openai
import streamlit as st
import os

os.environ["STREAMLIT_HOME"] = "/tmp/.streamlit"

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

# Extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

# Parse the UPI transactions from the extracted text
def parse_transactions(text):
    pattern = r'(\d{2}/\d{2}/\d{4}) (Paid to|Received from) (.+?) INR([\d,]+\.\d{2})'
    matches = re.findall(pattern, text)
    data = []
    for date, txn_type, party, amount in matches:
        data.append({
            "Date": pd.to_datetime(date, dayfirst=True),
            "Transaction Type": txn_type,
            "Party": party,
            "Amount (INR)": float(amount.replace(',', ''))
        })
    return pd.DataFrame(data)

# Analyze and summarize the data
def generate_summary(df):
    total_spent = df[df["Transaction Type"] == "Paid to"]["Amount (INR)"].sum()
    total_received = df[df["Transaction Type"] == "Received from"]["Amount (INR)"].sum()
    top_expenses = df[df["Transaction Type"] == "Paid to"].groupby("Party")["Amount (INR)"].sum().sort_values(ascending=False).head(3)
    return {
        "Total Spent": total_spent,
        "Total Received": total_received,
        "Top Expenses": top_expenses.to_dict()
    }

# Get recommendations using OpenAI
def get_recommendations(summary):
    prompt = f"""
    Here is my UPI transaction summary:
    Total Spent: ₹{summary['Total Spent']}
    Total Received: ₹{summary['Total Received']}
    Top Expenses: {summary['Top Expenses']}
    
    Based on this, please give me personalized financial advice, tips to save money, and monthly budgeting suggestions.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("💳 Personal UPI Usage & Financial Analyzer")

uploaded_file = st.file_uploader("📤 Upload your UPI PDF Statement", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Extracting and analyzing your data..."):
        text = extract_text_from_pdf(uploaded_file)
        df = parse_transactions(text)
        if df.empty:
            st.warning("No transactions found in this PDF format.")
        else:
            st.success("✅ Transactions parsed successfully!")
            st.subheader("📊 Transaction Table")
            st.dataframe(df)

            summary = generate_summary(df)
            st.subheader("📌 Financial Summary")
            st.write(f"**Total Spent:** ₹{summary['Total Spent']}")
            st.write(f"**Total Received:** ₹{summary['Total Received']}")
            st.write("**Top 3 Expense Categories:**")
            for party, amt in summary['Top Expenses'].items():
                st.write(f"- {party}: ₹{amt}")

            if st.button("🧠 Get Financial Advice"):
                advice = get_recommendations(summary)
                st.subheader("💡 Personalized Advice")
                st.write(advice)
