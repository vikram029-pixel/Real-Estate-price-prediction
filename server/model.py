#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns

#Reading file
df=pd.read_csv('Bengaluru_House_Data.csv')
df.head()

#Removing features
df.drop(['society','balcony','availability','area_type'],inplace=True,axis=1)
df.head()

#Data cleaning
df.isnull().sum()

df.shape

#Removing null values
df2=df.dropna()
df2.head()


df2['size'].unique()

df2['bhk']=df2['size'].apply(lambda x:int(x.split(' ')[0]))
df2.head()

df2.total_sqft.unique()

#resolving the range values & other type values in total_sqft
def range_sol(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0]) + float(token[1]))/2
    
    try:
        return float(x)
    except:
        return None

df3=df2.copy()    
df3['total_sqft']=df3['total_sqft'].apply(range_sol)
#df3.loc[30]

## feature engineering & dimension reduction
df3['price_per_sqft']=df3['price']*100000/df3['total_sqft']
df3.head()

len(df3.location.unique())

df3.location=df3.location.apply(lambda x:x.strip())
location_stats=df3.groupby('location')['location'].count().sort_values(ascending=False)
len(location_stats[location_stats<=10])

location_stats_less_than_10=location_stats[location_stats<=10]
df3.location=df3.location.apply(lambda x:'others'if x in location_stats_less_than_10 else x)
len(df3.location.unique())

## Removing outliers
# let us consider min size of room to be 300 sqft , if something lies below it will be consider a outlier.

df3[df3.total_sqft/df3.bhk < 300].head(3)
df4=df3[~(df3.total_sqft/df3.bhk < 300)]
df4.shape

# if No.of bathroom is 2 > No.of BHK then it will be a outlier
df4=df4[df4.bath<df4.bhk+2]
df4.shape

# around 68% points lies btw mean & 1 std deviation

def remove_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        std=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-std)) & (subdf.price_per_sqft<=(m+std))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out    
        
df5=remove_outliers(df4)
df5.shape

#property price smaller bhk cann't be > price for bigger bhk for same area

# def plot_scatter_chart(df,location):
#     bhk2=df[(df.location==location) & (df.bhk==2)]
#     bhk3=df[(df.location==location) & (df.bhk==3)]
#     plt.rcParams['figure.figsize']=(8,5)
#     plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2bhk',s=40)
#     plt.scatter(bhk3.total_sqft,bhk3.price,color='green',marker='+',label='3bhk',s=40)
#     plt.xlabel('total_sqft')
#     plt.ylabel('price_per_sqft')
#     plt.title(location)
#     plt.legend()

# plot_scatter_chart(df5,'Rajaji Nagar') 

def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df6=remove_bhk_outliers(df5)
df6.shape

# plot_scatter_chart(df6,'Rajaji Nagar')

df6.drop(['size','price_per_sqft'],axis=1,inplace=True)
df6.head(3)

### converting text into dummies

dummies=pd.get_dummies(df6.location)
df7=pd.concat([df6,dummies.drop('others',axis='columns')],axis='columns')
df7.drop('location',axis=1,inplace=True)
df7.head(3)

## model building

X=df7.drop('price',axis=1)
y=df7['price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X_train,y_train)
lm.score(X_test,y_test)

#shuffle fold method
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
cross_val_score(LinearRegression(),X,y,cv=cv)

#supplying outside data
def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
        
    return lm.predict([x])[0]    

