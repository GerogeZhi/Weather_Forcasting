import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
data=pd.read_table('wuhan.txt',sep=' ')

'''
等待API就位
import requests
hello=requests.get('https://api.openweathermap.org/data/2.5/forecast?id=1791247&APPID=9c15e03b02f4171f80d4320c2a483f34')
data=hello.json()
data.keys()
data['city']
data['list'][0]
data['list'][0].keys()
data['list'][0]['weather']

wea=pd.Series()
for i in range(len(data['list'])):
    y=pd.DataFrame(data['list'][i]['weather'])
    wea=pd.concat([wea,y],ignore_index=True)
wea=wea.drop(wea.columns[0],axis=1)
 
'''
  
#数据预处理
dddd=data.index
data.insert(0,'ssss',dddd)
data.columns
data.columns=['Station_Id_C', 'Year_Data', 'Mon_Data', 'Day_Data', 'Hour_Data', 'PRS',
       'PRS_Sea', 'WIN_D', 'WIN_S', 'TEM', 'RHU', 'PRE_1h','leftone']
data=data.drop(['leftone'],axis=1)
data.index=range(len(data.index))


tslice=data[(data.Day_Data<=10) & (data.Day_Data>8)]
plt.figure()
plt.plot(tslice.Hour_Data[:8],tslice.TEM[:8],'ro-',linewidth=0.5,markersize=5,label='day nine')
plt.plot(tslice.Hour_Data[8:],tslice.TEM[8:],'gs-',linewidth=0.5,markersize=5,label='day ten')
plt.xlabel('hours')
plt.ylabel('temp')
plt.axis([0,20,0,40])
plt.legend(loc='upper right')


  
weather=data.iloc[:,4:]
sns.pairplot(weather)


#转到以天为单位（直接暴力求均值）
some=[]
for i in range(19):
    some.append(data.iloc[i*8:(i+1)*8].mean())
shit=pd.DataFrame(some)

plt.plot(shit.TEM,'ro-',linewidth=0.5,markersize=5,label='weather tem')
plt.xlabel('time_day')
plt.ylabel('tempture')
plt.axis([0,20,0,40])
plt.legend(loc='upper right')

        
#data.iloc[].groupby(data.Day_data)


'''
plt.pcolor(weather)
corr_weather=pd.DataFrame.corr(weather.transpose())
plt.pcolor(corr_weather)
from sklearn.cluster.bicluster import SpectralCoclustering
model=SpectralCoclustering(n_clusters=6,random_state=0)
model.fit(corr_weather)
model.rows_
np.sum(model.rows_,axis=1)
np.sum(model.rows_,axis=0)
model.row_labels_
'''

#交互图
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import show,output_file
# edit this to make the function `location_plot`.
def location_plot(title,colors):
    output_file(title+".html")
    location_source = ColumnDataSource(
        data={
            "x": shit["Day_Data"],
            "y": shit["TEM"],
            "colors": colors,
            "RHU": shit.RHU,
            "Month": shit.Mon_Data
        }
    )

    fig = figure(title = title,
        x_axis_location = "above", tools="resize, hover, save")
    fig.plot_width  = 750
    fig.plot_height = 300
    fig.circle("x", "y", size=9, source=location_source,
         color='colors', line_color = None)
    fig.xaxis.major_label_orientation = np.pi / 3
    hover = fig.select(dict(type = HoverTool))
    hover.tooltips = {
        "Month & day": "(@Month , @x)",
        "tempture":"@y",
        "Humidity": "@RHU"    
    }
    show(fig)
    
region_cols = ["red", "orange", "green", "blue", "purple", "gray"]
location_plot("that's it", region_cols)



'''
#ready to boosting
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation

train_df =
test_df=

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self,X,y=None):
        self.fill=pd.Series([X[c].value_counts().index[0] 
            if X[c].dtype==np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self,X,y=None):
        return X.fillna(self.fill)
    
feature_columns_to_use=['..............']
nonnumeric_columns=['.........']
big_X=train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed=DataFrameImputer().fit_transform(big_X)
le=LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature]=le.fit_transform(big_X_imputed[feature])
train_X=big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X=big_X_imputed[train_df.shape[0]::].as_matrix()
train_y=train_df['Survived']

gbm=xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05).fit(train_X,train_y)

scores=cross_validation.cross_val_score(gbm,train_X,train_y,cv=8)
predictions=gbm.predict(test_X)
'''
