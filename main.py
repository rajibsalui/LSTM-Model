import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import yfinance as yf

start = '2010-01-01'
end = '2023-12-31'

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010 - 2023')
st.write(df.describe())

#visualizations
st.subheader('Closing Price vs Time chart')
ma100=df.Close.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 days Moving average')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #70% data
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])  #30% data

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



model=load_model.load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original data')
y_predicted_2d = y_predicted[:, -1, 0]  

fig2=plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted_2d, 'r', label='Predicted Price')  # Use the reshaped array
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original vs Predicted Prices')
st.pyplot(fig2)