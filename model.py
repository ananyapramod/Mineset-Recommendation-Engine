import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
minedata=pd.read_csv("C:\\Users\\canara\\Documents\\MineData.csv")
airdata=pd.read_csv("C:\\Users\\canara\\Documents\\air quality dataset sample.csv ")
waterdata=pd.read_csv("C:\\Users\\canara\\Documents\\Surface Water Quality Analysis.csv")





airval=airdata.columns.values
x=waterdata['pH'].quantile(0.90)
y=airdata['PM10'].quantile(0.90)
indexvalue=airdata['PM10']/500+airdata['PM2.5']/500 +airdata['NO2']/500 + airdata['O3']/1000+airdata['CO']/50 +airdata['SO2']/2000 +airdata['NH3']/2000+airdata['Pb']/50
indexvalue=indexvalue*1000/8

a=airdata[['Time','Location']]
airindex=pd.concat([a,indexvalue],axis=1)
airindex.columns=["Time","Location","Air Quality Index"]
print("Enter the location to predict for:")
#i=input()
i="B"
print(i)
data=airindex.copy().loc[airindex['Location']==i]


X=pd.DataFrame([i for i in range(1,len(data)+1)])
Y=data['Air Quality Index'].reset_index(drop=True)
Z=pd.concat([X,Y],axis=1)
model=LinearRegression().fit(X,Y)

x=model.predict([[6]])
print("The air quality index for location" + str(i) +" will be predicted to be "+str(x))