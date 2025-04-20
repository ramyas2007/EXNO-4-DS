# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
REG NO: 212224040268

NAME  : RAMYA S
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/46d6d289-c5f1-4f31-adac-0b03188739fe)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/df453fa7-10f8-4865-9a0c-584c3d228a12)
```
max_val=np.max(np.abs(df[['Height','Weight']]))
max_val
```

![image](https://github.com/user-attachments/assets/974ae56a-37ef-42fd-aa9d-d4940947acdb)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/31077882-ac5a-4b01-a89a-aae4d5101026)
```
from sklearn.preprocessing import Normalizer
nm=Normalizer()
df[['Height','Weight']]=nm.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/63b5e19e-0eea-47b3-9060-111485f771cc)
```
from sklearn.preprocessing import MaxAbsScaler
mas=MaxAbsScaler()
df[['Height','Weight']]=mas.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/3c0e2181-fe26-4963-924b-4e53fe212459)
```
from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
df[['Height','Weight']]=rs.fit_transform(df[['Height','Weight']])
df.head(5)
```
![image](https://github.com/user-attachments/assets/f28e85e4-5c10-4b71-a3cd-38c865dc5874)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/97091902-df55-4571-8fbe-8d1d6da034db)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
contingency_table
```
![image](https://github.com/user-attachments/assets/bf8ed155-d73d-4934-9577-5ff566fe3ee5)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print('Chi-square statistic:',chi2)
print('p-value:',p)
```
![image](https://github.com/user-attachments/assets/70ecc174-0746-4a67-bcbe-dc93b460fb56)
```
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
'Feature1' : [1,2,3,4,5],'Feature2' : ['A','B','C','A','B'],'Feature3' : [0,1,1,0,1],'Target': [0,1,1,0,1]}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/183b8adf-af0c-4635-91ea-ea588908e9a4)
```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
print('Selected features:',x_new)
```
![image](https://github.com/user-attachments/assets/552c6986-5636-4c85-bb29-959672154030)
```
selectedFeatureIndices=selector.get_support(indices=True)
selectedFeatures=x.columns[selectedFeatureIndices]
print('Selected features:',selectedFeatures)
```
![image](https://github.com/user-attachments/assets/31c1f21c-fd34-4890-9200-bf51765ad0eb)

# RESULT:
Feature Scaling and Feature Selection process has been successfully performed on the data set.
