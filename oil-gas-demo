import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initial page configuration
st.set_page_config(page_title='Oil and Gas Equipment Dashboard', layout='wide')
st.title('Analysis and Prediction of Oil and Gas Equipment Data')

# Load sample dataset from CSV or other sources
@st.cache_data
def load_data():
    # Replace 'equipment_data.csv' with the actual path to the dataset
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/oil_equipment_data.csv'  # Example link, change as needed
    data = pd.read_csv(url)
    return data

data = load_data()
st.write("### Initial Data:")
st.dataframe(data.head())

# Basic data analysis
st.write("### Statistical Overview:")
st.write(data.describe())

# Display charts for data analysis
st.write("### Analytical Charts:")
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data['Price'], bins=30, ax=ax[0], kde=True)
ax[0].set_title('Price Distribution of Equipment')
sns.countplot(x='Category', data=data, ax=ax[1])
ax[1].set_title('Number of Equipment per Category')
st.pyplot(fig)

# Prepare data for modeling
st.write("### Data Preparation for Modeling:")
data = data.dropna(subset=['Age', 'Price'])
X = data[['Age', 'Category']]
y = data['Price']

# Convert categorical data to numerical values
X = pd.get_dummies(X, columns=['Category'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display model results
st.write("### Model Results:")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write("Coefficients:")
st.write(pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficient']))

# Predict with new input data
st.write("### Prediction Based on New Input Data:")
age_input = st.slider('Equipment Age:', min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=10)
category_input = st.selectbox('Equipment Category:', options=sorted(data['Category'].unique()))

# Prepare new input for prediction
input_data = pd.DataFrame([[age_input] + [1 if cat == category_input else 0 for cat in sorted(data['Category'].unique())[1:]]], columns=X.columns)
predicted_price = model.predict(input_data)[0]

st.write(f"The predicted price for equipment with age {age_input} and category {category_input} is: ${predicted_price:.2f}")
