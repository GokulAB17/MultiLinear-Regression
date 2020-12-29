#Predict Price of the computer
#A dataframe containing :
#price : price in US dollars of 486 PCs
#speed : clock speed in MHz
#hd : size of hard drive in MB
#ram : size of Ram in in MB
#screen : size of screen in inches
#cd : is a CD-ROM present ?
#multi : is a multimedia kit (speakers, sound card) included ?
#premium : is the manufacturer was a "premium" firm (IBM, COMPAQ) ?
#ads : number of 486 price listings for each month
#trend : time trend indicating month starting from January of 1993 to November of 1995.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading the data
comp=pd.read_csv(r"provide path\Computer_Data.csv")

#to view columns 
comp.columns

#Data Preprocessing


#Mapping values for col "cd","multi","premium" to dummy variables for ease of operation 
comp.cd=comp.cd.map({"yes":1,"no":0})
comp.multi=comp.multi.map({"yes":1,"no":0})
comp.premium=comp.premium.map({"yes":1,"no":0})
comp

#Removing unwanted columns
comp1=comp.drop(["Unnamed: 0"],axis=1)

#Exploratory Data analysis
#correlation
corr=comp1.corr()
#no collinearity problem between input variables

## Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(comp1)

#Building Model
import statsmodels.formula.api as smf

m1=smf.ols("price~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=comp1).fit()
m1.params
m1.summary()

#Rsq--0.776
#pvalues less than 0.05

import statsmodels.api as sm
sm.graphics.influence_plot(m1)
#no influence points

#to improve Rsq value we ll perform transformations
m1_exp=smf.ols("np.log(price)~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=comp1).fit()
m1_exp.params
m1_exp.summary()
#Rsq--0.783

m1_quad=smf.ols("price~speed*speed+hd*hd+ram*ram+screen*screen+cd*cd+multi*multi+premium*premium+ads*ads+trend*trend",data=comp1).fit()
m1_quad.summary()
#Rsq--0.776

#we select exponential as final
final_m=smf.ols("np.log(price)~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=comp1).fit()
final_m.params
final_m.summary()

#prediction of price with final model
price_predict=np.exp(final_m.predict(comp1))

# Observed values VS Fitted values
plt.scatter(comp1.price,price_predict,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
price_predict.corr(comp1.price)
#high collinearity value 0.883
# Residuals VS Fitted Values 
plt.scatter(price_predict,final_m.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

np.mean(final_m.resid_pearson)
#mean value 1.965e-13 ===0
#assumptions for errors/residuals satisfied

# histogram
plt.hist(final_m.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_m.resid_pearson, dist="norm", plot=pylab)


## Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
price_train,price_test  = train_test_split(comp1,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("np.log(price)~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=price_train).fit()

# train_data prediction
train_pred = np.log(model_train.predict(price_train))

# train residual values  err/residual = Predicted value - Actual Value
train_resid  = train_pred - price_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = np.log(model_train.predict(price_test))

# test residual values 
test_resid  = test_pred - price_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
