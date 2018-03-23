import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Find nearest cities

location = pd.read_excel('China_SURF_Station.xlsx')
location.head()
hubei=location[location['省份']=='湖北']
hubei=hubei.reset_index(drop=True)
wuhan=hubei[hubei['站名'] == '武汉']

wuhanJ=np.array(wuhan['经度'])
wuhanW=np.array(wuhan['纬度'])
hubeiJ=np.array(hubei['经度'])
hubeiW=np.array(hubei['纬度'])

real=np.zeros(hubei.shape[0])
for i in range(len(hubei)):
    real[i]=np.power(hubeiJ[i]-wuhanJ,2)+np.power(hubeiW[i]-wuhanW,2)
    distances=np.sqrt(np.array(real))
distances = pd.Series(distances)

cities_fake_D = distances[distances < distances.mean()]
cities_near_D = cities_fake_D[cities_fake_D < cities_fake_D.mean()]
cities = hubei.loc[cities_near_D.index]
cities['distances']=cities_near_D
cities = cities.reset_index(drop=True)
cities = cities.loc[np.argsort(cities['distances'])]
cities = cities.reset_index(drop=True)

num=np.array(cities['区站号'])
np.sort(num)

# load data

# from download

weather=pd.read_excel('hubei_data.xlsx')
wuhan_W = weather[weather['Station_Id_C']==57494]
others = weather[weather['Station_Id_C']!=57494]

# load Nearst cities' data

cities_near=np.zeros((1,29))
for s in num:
    cities_near = np.concatenate((cities_near,np.array(others[others['Station_Id_C']==s])),axis=0)
cities_N = pd.DataFrame(cities_near)
cities_Near = cities_N.drop(0)
cities_Near.columns = weather.columns


# select predictors

from sklearn.feature_selection import SelectKBest,f_classif
predictors_full = ['PRS', 'PRS_Sea', 'PRS_Max', 'PRS_Min', 'WIN_S_Max', 'WIN_S_Inst_Max',
       'WIN_D_INST_Max', 'WIN_D_Avg_2mi', 'WIN_S_Avg_2mi', 'WIN_D_S_Max',
       'TEM', 'TEM_Max', 'TEM_Min', 'RHU', 'VAP', 'RHU_Min', 'PRE_1h', 'VIS',
       'CLO_Cov', 'CLO_COV_LM', 'CLO_Cov_Low', 'windpower','tigan']

selector=SelectKBest(f_classif,k=5)
selector.fit(weather[predictors_full],weather['WEP_Now'])
scores=-np.log10(selector.pvalues_)

plt.bar(range(len(predictors_full)),scores)
plt.xticks(range(len(predictors_full)),predictors_full,rotation='vertical')

predictors = ['PRS', 'PRS_Sea', 'PRS_Max', 'PRS_Min', 'WIN_S_Max', 'WIN_S_Inst_Max','windpower',
              'WIN_D_Avg_2mi', 'WIN_S_Avg_2mi', 'VAP', 'PRE_1h', 'CLO_Cov', 'CLO_Cov_Low', 'tigan']

# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
alg=RandomForestClassifier(random_state=1,n_estimators=200,min_samples_split=8,min_samples_leaf=6)

scores=cross_validation.cross_val_score(alg,others[predictors_full],others['WEP_Now'],cv=100)
scores.mean()

#0.9002 hubei all cities except wuhan

scores=cross_validation.cross_val_score(alg,weather[predictors_full],weather['WEP_Now'],cv=100)
scores.mean()

# 0.8970 hubei all cities

scores=cross_validation.cross_val_score(alg,cities_Near[predictors_full],cities_Near['WEP_Now'],cv=50)
scores.mean()

#0.8915 Near cities

kf=cross_validation.KFold(weather.shape[0],n_folds=200,random_state=1)
predictions=[]   
for train,test in kf:
    alg.fit(others[predictors_full].iloc[train,:],others['WEP_Now'].iloc[train])
    predictions.append(alg.predict(others[predictors_full].iloc[test,:]))
predictions=np.concatenate(predictions,axis=0)
predictions = pd.Series(predictions)

#WEP = weather['WEP_Now'].reset_index(drop=True)
WEP = others['WEP_Now'].reset_index(drop=True)
sum(predictions==WEP)/len(WEP)

#0.8800 Near cities by cross_validation

answers = alg.predict(wuhan_W[predictors_full])
sun = wuhan_W['WEP_Now'].reset_index(drop=True)
sum(answers==sun)/len(sun)

#0.6734 by near cities
#0.6734 by other cities

plt.plot(answers,'ro-',linewidth=0.5,markersize=3)
plt.plot(sun,'bo-',linewidth=0.5,markersize=2)








#from API

import requests
hello_wuhan = requests.get('http://api.data.cma.cn:8090/api?userId=521513276064vKF0R&pwd=rGCoBUX&dataFormat=json&interfaceId=getSurfEleByTimeRangeAndStaID&dataCode=SURF_CHN_MUL_HOR&timeRange=[20180322000101,20180322100000]&staIDs=57494&elements=Station_Id_C,Year,Mon,Day,Hour,PRS,PRS_Sea,PRS_Max,PRS_Min,TEM,TEM_Max,TEM_Min,RHU,RHU_Min,VAP,PRE_1h,WIN_D_INST_Max,WIN_S_Max,WIN_D_S_Max,WIN_S_Avg_2mi,WIN_D_Avg_2mi,WEP_Now,WIN_S_Inst_Max')

wuhan=hello_wuhan.json()
wuhan.keys()
data=pd.DataFrame(wuhan['DS'])
weather = data.iloc[:,3:22]
weather = weather.drop('Station_Id_C',axis=1)

# capsule

plt.plot(weather['WEP_Now'],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(0,169,24))

plt.figure(figsize=(10,8))
plt.subplot(221)
plt.plot(weather['WEP_Now'][:49],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(0,49,8))
plt.subplot(222)
plt.plot(weather['WEP_Now'][48:97],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(48,97,8))
plt.subplot(223)
plt.plot(weather['WEP_Now'][96:145],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(96,145,8))
plt.subplot(224)
plt.plot(weather['WEP_Now'][144:169],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(144,169,8))
 


# feature_selection

from sklearn.feature_selection import SelectKBest,f_classif

predictors_full = ['PRS','PRS_Sea','PRS_Max','PRS_Min','TEM',
                   'TEM_Max','TEM_Min','RHU','RHU_Min','VAP',
                   'PRE_1h','WIN_D_INST_Max','WIN_S_Max','WIN_D_S_Max',
                   'WIN_S_Avg_2mi','WIN_D_Avg_2mi','WIN_S_Inst_Max']

selector=SelectKBest(f_classif,k=5)
selector.fit(weather[predictors_full],weather['WEP_Now'])
scores=-np.log10(selector.pvalues_)

plt.bar(range(len(predictors_full)),scores)
plt.xticks(range(len(predictors_full)),predictors_full,rotation='vertical')


predictors=['PRE_1h','RHU','RHU_Min','TEM','TEM_Max','TEM_Min','WIN_D_INST_Max','WIN_D_S_Max','VAP']




from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
alg=RandomForestClassifier(random_state=1,n_estimators=40,min_samples_split=6,min_samples_leaf=4)
scores=cross_validation.cross_val_score(alg,weather[predictors_full],weather['WEP_Now'],cv=8)
scores.mean()
# 0.667

kf=cross_validation.KFold(weather.shape[0],n_folds=8,random_state=1)
predictions=[]   
for train,test in kf:
    alg.fit(weather[predictors].iloc[train,:],weather['WEP_Now'].iloc[train])
    predictions.append(alg.predict(weather[predictors].iloc[test,:]))
predictions=np.concatenate(predictions,axis=0)
predictions = pd.Series(predictions)
corrects=np.sum(predictions==weather['WEP_Now'])
corrects/len(weather)

#0.625

plt.plot(predictions[:48],'ro-',linewidth=0.5,markersize=3)
plt.plot(weather['WEP_Now'][:48],'bo-',linewidth=0.5,markersize=2)

plt.plot(predictions[48:97],'ro-',linewidth=0.5,markersize=3)
plt.plot(weather['WEP_Now'][48:97],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(48,97,8))

plt.plot(predictions[96:145],'ro-',linewidth=0.5,markersize=3)
plt.plot(weather['WEP_Now'][96:145],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(96,145,8))

plt.plot(predictions[144:169],'ro-',linewidth=0.5,markersize=3)
plt.plot(weather['WEP_Now'][144:169],'bo-',linewidth=0.5,markersize=2)
plt.xticks(range(144,169,8))

# Try SVM

from sklearn import svm
alg=svm.SVC()
scores=cross_validation.cross_val_score(alg,others[predictors],others['WEP_Now'],cv=50)
scores.mean()
#0.6276


# Try Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

kf=KFold(weather.shape[0],n_folds=3,random_state=1)
alg=LinearRegression()

predictors=['PRE_1h','RHU','RHU_Min','TEM','TEM_Max','TEM_Min','WIN_D_INST_Max','WIN_D_S_Max','VAP']

predictions=[]   
for train,test in kf:
    alg.fit(weather[predictors].iloc[train,:],weather['WEP_Now'].iloc[train])
    predictions.append(alg.predict(weather[predictors].iloc[test,:]))

predictions=np.concatenate(predictions,axis=0)

predictions = predictions.astype(int)

'''
accuracy=len(predictions[predictions==weather['WEP_Now']])/len(predictions)
'''
# linear regression pass

# cross_validation

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

alg=LogisticRegression(random_state=1)

scores=cross_validation.cross_val_score(alg,weather[predictors],weather['WEP_Now'],cv=kf)
scores.mean()

# 0.434

'''
alg=LogisticRegression(random_state=1)
alg.fit(weather[predictors],weather['WEP_Now'])
predictionsLR=alg.predict(weather_test[predictors])
'''



from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

algorithms=[[GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3),predictors_full],
        [LogisticRegression(random_state=1),predictors]
    ]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=30, max_depth=4), predictors_full],
    [LogisticRegression(random_state=1), predictors]
   ]

kf=KFold(weather.shape[0],n_folds=3,random_state=1)
predictions=[]
for train,test in kf:
    full_predictions=[]
    for alg,predictors in algorithms:
        alg.fit(weather[predictors].iloc[train,:],weather['WEP_Now'].iloc[train])
        test_predictions=alg.predict_proba(weather[predictors].iloc[test,:].astype(float))[:,1]
        full_predictions.append(test_predictions)
    test_predictions=(full_predictions[0]+full_predictions[1])/2
    predictions.append(test_predictions)
predictions=np.concatenate(predictions)
len(predictions[predictions>0.4])

# 102 / 176 = 0.5795
'''  
accuracy=len(predictions[predictions==weather['WEP_Now']])/len(predictions) 
'''















