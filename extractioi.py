# app.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.title("📊 Transaction Categorization Dashboard")

# Sample DataFrame (replace this with actual extracted transaction data)
data = {
    'Description': ['Swiggy order', 'Amazon purchase', 'Uber ride', 'Bigbasket groceries', 'Recharge prepaid', 'Received salary'],
    'Amount': [150, 200, 300, 100, 50, 500]
}

# Create the DataFrame
df = pd.DataFrame(data)
df.columns = df.columns.str.strip()

# 🔍 Categorization function
def categorize_transaction(description):
    description = description.lower()
    if "swiggy" in description or "zomato" in description:
        return "Food"
    elif "amazon" in description or "flipkart" in description:
        return "Shopping"
    elif "uber" in description or "ola" in description:
        return "Transport"
    elif "bigbasket" in description:
        return "Groceries"
    elif "recharge" in description:
        return "Utilities"
    elif "cashback" in description:
        return "Rewards"
    elif "received" in description:
        return "Income"
    else:
        return "Others"

# Apply categorization
df['Category'] = df['Description'].apply(categorize_transaction)

# --- Data Quality Check ---
st.subheader("🧼 Data Quality Check")
st.write("Missing values:")
st.write(df.isnull().sum())

df = df.drop_duplicates()

# --- Data Preview ---
st.subheader("📋 Sample Transactions")
st.dataframe(df.head())

# --- Pie Chart: Category Distribution ---
st.subheader("📊 Transaction Category Distribution")
category_counts = df['Category'].value_counts()

fig1, ax1 = plt.subplots()
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax1)
ax1.set_ylabel('')
ax1.set_title("Category Distribution")
st.pyplot(fig1)

# --- Histogram: Amount ---
st.subheader("💰 Transaction Amount Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['Amount'], kde=True, ax=ax2)
ax2.set_title("Amount Distribution")
st.pyplot(fig2)

# --- Correlation Heatmap ---
st.subheader("🔗 Correlation Heatmap")
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

fig3, ax3 = plt.subplots()
sns.heatmap(df[['Amount', 'Category_encoded']].corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

# --- Model Building ---
st.subheader("🤖 Category Prediction Model (Random Forest)")

X = df[['Amount']]
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.write("### Classification Report")
st.code(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))
