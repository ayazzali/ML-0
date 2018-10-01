
# coding: utf-8

# In[1]:


def l(*t):
    print(t)
def ac(model,y_valid,X_valid ):
    print(accuracy_score(y_valid.tolist(),model.predict(X_valid)))
def dist(x1,y1,x2,y2):
    return np.linalg.norm(x1-x2) + np.linalg.norm(y2-y1)

'17 раз до 9 27сек'
'100 3.3 3мин'
import pandas as pd
import numpy as np
import datetime 
import matplotlib.pyplot as plt


# In[2]:


def readCsv(path,nrows=10000):
    data=pd.read_csv(path,
                 #dtype=types,
                 nrows=nrows,
                 parse_dates=['pickup_datetime',
                              #'dropoff_datetime',
                              
                             ],
                 date_parser=lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z'),
                 usecols=
                         ['vendor_id',
                         'pickup_datetime',
                         #'dropoff_datetime',
                         'pickup_longitude',
                         'pickup_latitude',
                         'dropoff_longitude',
                         'dropoff_latitude',
                         'rate_code',
                         'passenger_count',
                         'trip_distance',

                         'fare_amount',
                         'tolls_amount',
                        ])
    return data


# In[3]:


import glob
def csvs(pathmatch,nrows=10000):
    r=pd.DataFrame()
    for path in glob.glob(pathmatch):
        print(' reading '+path)
        ir=readCsv(path,nrows)
        #ir=pd.read_csv(path,nrows=nrows,date_parser=lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z'))
        r=r.append(ir ,ignore_index=True)
        print(len(r))
        if(len(r)>=nrows):
            print('csvs return')
            break
    return r
        
r=csvs('qwe0*',1000)
r[:1]


# In[ ]:





# In[4]:


r.info()


# In[5]:


get_ipython().run_cell_magic('time', '', "q=r.pickup_datetime.apply(pd.to_datetime)\n#q=r.pickup_datetime.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z'))")


# In[6]:


get_ipython().run_cell_magic('time', '', "q=r.pickup_datetime.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S %Z'))")


# In[7]:


data=readCsv('qwe000000000000.csv',nrows=100)
data.info()
#datetime.datetime.strptime('2010-03-04 00:35:16','%a %b %d %H:%M:%S %Y')
#datetime.datetime.strptime('2010-03-04 00:35:16 UTC','%Y-%m-%d %H:%M:%S %Z')


# In[8]:


def create_features(sourcedata):
    data=sourcedata.copy()
    
    # create fare
    data['fare']=data['tolls_amount']+data['fare_amount']
    data=data.drop('tolls_amount',axis=1)
    data=data.drop('fare_amount',axis=1)
    
    # filter
    data=data[data['fare']>0]
    len(data[data['fare']>0])
    data=data[data.trip_distance>0]
    data=data[(data.pickup_longitude>-80)&(data.pickup_longitude<-20)] # -77
    data=data[(data.dropoff_longitude>-80)&(data.dropoff_longitude<-20)] # -77
    data=data[(data.dropoff_latitude>10)&(data.dropoff_latitude<60)] # 40
    data=data[(data.pickup_latitude>10)&(data.pickup_latitude<60)] # 40
    data=data[ data.rate_code.isnull()==False]
    #data=data[data['rate_code']!=np.nan]
    
    #
    data['p_hour']=data.pickup_datetime.map(lambda x: x.hour)
    data['p_dayofweek']=data.pickup_datetime.map(lambda x: x.dayofweek)
    data.drop('pickup_datetime',axis=1,inplace=True)
    
    # round
    data.pickup_latitude=data.pickup_latitude.apply(lambda x: np.round(x,2))#.apply(np.str)
    data.pickup_longitude=data.pickup_longitude.apply(lambda x: np.round(x,2))
    data.dropoff_latitude=data.dropoff_latitude.apply(lambda x: np.round(x,2))
    data.dropoff_longitude=data.dropoff_longitude.apply(lambda x: np.round(x,2))
    
    
    return data
data2=create_features(data)
data2


# In[9]:



plt.plot(data.pickup_datetime)
#[q for q in data.pickup_datetime if q<pd.datetime(1920,10,10) ]

#data3=data2[(data2.pickup_longitude>-21)]

#dist(x1,y1,x2,y2):
#[dist(x.pickup_longitude,x.dropoff_longitude,1,1) for x in data]
#[x for x in data2.values]


# In[10]:


data2.describe()


# In[11]:


# dists=[dist(x[1],x[2],x[3],x[4]) for x in data2.values]
# plt.plot(dists)
# plt.plot(data2.trip_distance,color='red')
# dd=pd.DataFrame(dists)

