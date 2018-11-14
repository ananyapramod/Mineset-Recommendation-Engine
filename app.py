import pandas as pd
from datetime import timedelta
from datetime import datetime

#import data
minedata=pd.read_csv("C:\\Users\\canara\\Documents\\MineData.csv")
airdata=pd.read_csv("C:\\Users\\canara\\Documents\\air quality dataset sample.csv ")
waterdata=pd.read_csv("C:\\Users\\canara\\Documents\\Surface Water Quality Analysis.csv")
noisedata=pd.read_csv("C:\\Users\\canara\\Documents\\noisedata.csv")

#air
airval=airdata.columns.values
airthreshold=airdata['PM10'].quantile(0.90) + airdata['PM2.5'].quantile(0.90)+airdata['NO2'].quantile(0.90) + airdata['O3'].quantile(0.90)+airdata['CO'].quantile(0.90) +airdata['SO2'].quantile(0.90) +airdata['NH3'].quantile(0.90)+airdata['Pb'].quantile(0.90)
airindex=airdata['PM10']/500+airdata['PM2.5']/500 +airdata['NO2']/500 + airdata['O3']/1000+airdata['CO']/50 +airdata['SO2']/2000 +airdata['NH3']/2000+airdata['Pb']/50
airindex=airindex*1000/8
airthreshold=airthreshold/8


a=airdata[['Time','Location']]
airindex=pd.concat([a,airindex],axis=1)
airindex.columns=["Time","Location","Air Quality Index"]

print("The inspection of the following mines is recommended")

locations=['A','B','C','D']


recommenddata=airindex.copy().loc[airindex['Air Quality Index']>airthreshold]

uniqueloc=recommenddata.Location.unique()
#print("The inspection of the following mines is recommended:")
for i in uniqueloc:

    a = recommenddata.copy().loc[recommenddata['Location'] == i]
    maxval=a['Time'].max()
    print(i)
    print(a.copy().loc[a['Time']==maxval])



#water
waterval=waterdata.columns.values
waterthreshold=waterdata['pH'].quantile(0.80) + waterdata['BOD'].quantile(0.80)
waterindex=waterdata['pH']/10+waterdata['BOD']/10
waterindex=waterindex*10/2
waterthreshold=waterthreshold/2


a=waterdata[['Date','Location']]
waterindex=pd.concat([a,waterindex],axis=1)
waterindex.columns=["Date","Location","Water Quality Index"]


recommenddata=waterindex.copy().loc[waterindex['Water Quality Index']>waterthreshold]


uniqueloc=recommenddata.Location.unique()

for i in uniqueloc:
    a = recommenddata.copy().loc[recommenddata['Location'] == i]
    maxval=a['Date'].max()


    print(a.copy().loc[a['Date']==maxval])


noiseval=noisedata.columns.values
noisethreshold=noisedata['Decibel-Day'].quantile(0.90) + noisedata['Decibel-Night'].quantile(0.90)
noiseindex=noisedata['Decibel-Day']/10+noisedata['Decibel-Night']/10
noiseindex=noiseindex*10/2
noisethreshold=noisethreshold/2

a=noisedata[['Time','Location']]
noiseindex=pd.concat([a,noiseindex],axis=1)
noiseindex.columns=["Time","Location","Noise Index"]


locations=["A","B","C","D"]

recommenddata=noiseindex.copy().loc[noiseindex['Noise Index']>noisethreshold]


uniqueloc=recommenddata.Location.unique()

for i in uniqueloc:


    a=recommenddata.copy().loc[recommenddata['Location']==i]
    maxval = a['Time'].max()
    print(a.copy().loc[a['Time']==maxval])