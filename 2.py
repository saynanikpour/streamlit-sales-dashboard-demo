import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# تنظیمات اولیه صفحه
st.set_page_config(page_title='تحلیل فروش با هوش مصنوعی', layout='wide')
st.title('تحلیل و پیش‌بینی داده‌های فروش')

# بارگذاری دیتاست نمونه از فایل CSV یا Kaggle
@st.cache_data
def load_data():
    # به‌جای 'sales_data.csv' از مسیر واقعی دیتاست استفاده کنید
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'  # لینک مثال، تغییر دهید
    data = pd.read_csv(url)
    return data

data = load_data()
st.write("### داده‌های اولیه:")
st.dataframe(data.head())

# تحلیل اولیه داده‌ها
st.write("### اطلاعات آماری داده‌ها:")
st.write(data.describe())

# نمایش نمودارها برای تحلیل داده‌ها
st.write("### نمودارهای تحلیلی:")
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data['Fare'], bins=30, ax=ax[0], kde=True)
ax[0].set_title('توزیع قیمت بلیط')
sns.countplot(x='Pclass', data=data, ax=ax[1])
ax[1].set_title('تعداد مسافران در هر کلاس')
st.pyplot(fig)

# آماده‌سازی داده‌ها برای مدل‌سازی
st.write("### آماده‌سازی داده‌ها برای مدل‌سازی:")
data = data.dropna(subset=['Age', 'Fare'])
X = data[['Age', 'Pclass']]
y = data['Fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# نمایش نتایج مدل
st.write("### نتایج مدل:")
st.write(f"خطای میانگین مربعات: {mean_squared_error(y_test, y_pred):.2f}")
st.write("ضریب‌ها:")
st.write(pd.DataFrame(model.coef_, index=['Age', 'Pclass'], columns=['Coefficient']))

# پیش‌بینی با داده جدید
st.write("### پیش‌بینی بر اساس داده ورودی جدید:")
age_input = st.slider('سن مسافر:', min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=30)
pclass_input = st.selectbox('کلاس مسافر:', options=sorted(data['Pclass'].unique()))
predicted_fare = model.predict([[age_input, pclass_input]])[0]

st.write(f"پیش‌بینی قیمت بلیط برای مسافر با سن {age_input} و کلاس {pclass_input} برابر است با: ${predicted_fare:.2f}")
