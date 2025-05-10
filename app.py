import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Streamlit Title
st.title("UPI Transaction Extraction and Analysis")

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Process the uploaded file
    with pdfplumber.open(uploaded_file) as pdf:
        # Extract text from the first page (you can modify this to extract more pages or tables)
        page = pdf.pages[0]
        text = page.extract_text()
        if not text.strip():
            st.warning("No text found in the PDF.")
        else:
            st.write("Extracted Text from PDF:")
            st.write(text)

    # Example: Assuming you have extracted some data (mock example)
    # Example DataFrame for displaying UPI transactions
    data = {
        'Transaction ID': ['TX123', 'TX124', 'TX125', 'TX126'],
        'Description': ['Payment to ABC', 'Payment to XYZ', 'Refund from ABC', 'Payment to PQR'],
        'Amount': [500, 200, -150, 300],
        'Date': ['2025-05-10', '2025-05-09', '2025-05-08', '2025-05-07'],
    }

    df = pd.DataFrame(data)
    
    # Display the DataFrame
    st.write("Transaction Data:")
    st.dataframe(df)

    # Display a bar chart of transaction amounts
    st.subheader("Transaction Amounts")
    st.bar_chart(df['Amount'])

    # Visualizing data using Seaborn
    st.subheader("Seaborn Visualization (Amount Distribution)")
    sns.histplot(df['Amount'], kde=True)
    st.pyplot()

    # Machine Learning Example (Random Forest for Classification)
    st.subheader("Random Forest Example for Transaction Classification")
    
    # Convert 'Amount' to a categorical variable (positive -> 1, negative -> 0)
    df['Label'] = df['Amount'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df[['Amount']]  # Features
    y = df['Label']  # Target
    
    # Train a Random Forest model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Display the classification report
    st.write(f"Predictions: {predictions}")
    
    # Display model evaluation metrics
    report = classification_report(y_test, predictions)
    st.text(report)
