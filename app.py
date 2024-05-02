from keras.models import load_model
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date

model = load_model("PredictionModel.h5")

st.header("Stock Market Predictor")

stock=st.text_input("Enter Stock Symnbol","GOOG")
start="2000-01-01"
end=date.today().strftime("%Y-%m-%d")

data=yf.download(stock,start,end)

st.subheader("Stock Data")
st.write(data)
data_train=pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_test=pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

pas_100_days=data_train.tail(100)
data_test=pd.concat([pas_100_days,data_train],ignore_index=True)
data_test_scaler=scaler.fit_transform(data_test)

st.subheader("Price vs MA100 vs MA200")
ma_100_days=data.Close.rolling(100).mean()
ma_200_days=data.Close.rolling(200).mean()
fig1=plt.figure(figsize=(10,6))
plt.plot(data.Close,'b',label="Price")
plt.plot(ma_100_days,'r',label="MA 100")
plt.plot(ma_200_days,'g',label="MA 200")
plt.legend()
plt.show()
st.pyplot(fig1)

x=[]
y=[]
for i in range(100,data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])

x,y=np.array(x),np.array(y)

predict=model.predict(x)

scale=1/scaler.scale_
predict=predict*scale
y=y*scale

st.subheader("Original Price vs Pridicted Price")
fig2=plt.figure(figsize=(8,6))
plt.plot(predict[20:],'r',label="Predicted Price")
plt.plot(y,'g',label="Original Price")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
st.pyplot(fig2)






