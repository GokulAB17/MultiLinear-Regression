#Consider only the below columns and prepare a prediction model for predicting Price.

#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]


import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

# loading the dataset
#unicode issue encoding done in ASCII
c=pd.read_csv(r"provide_path\ToyotaCorolla.csv",encoding="unicode_escape")
c.columns

#Data preprocessing by reducing columns as per model requirement
corolla=pd.DataFrame(columns=["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"])

corolla.Price=c.Price
corolla.Age_08_04=c.Age_08_04
corolla.KM=c.KM
corolla.HP=c.HP
corolla.cc=c.cc
corolla.Doors=c.Doors
corolla.Gears=c.Gears
corolla.Quarterly_Tax=c.Quarterly_Tax
corolla.Weight=c.Weight

#Exploratory Data Analysis
corr=corolla.corr()
corr
#no collinearity between predictors

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(corolla)

#Model Building
import statsmodels.formula.api as smf
m1=smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=corolla).fit()
m1.params
m1.summary()
#Rsq=0.864
#only cc and doors have pvalue >0.05

m2=smf.ols("Price~cc",data=corolla).fit()
m2.summary()
#pvalue<0.05 it is significant

m3=smf.ols("Price~Doors",data=corolla).fit()
m3.summary()
#pvalue<0.05 it is significant

m4=smf.ols("Price~cc+Doors",data=corolla).fit()
m4.summary()
#pvalue<<0.05 it is signifiacnt

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(m1,cex=1.5)
#from graph we can see index no.80,960,221,601 are influencial 

corolla_new=corolla.drop(corolla.index[[80,221,960,601]],axis=0)

m_new=smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=corolla_new).fit()
m_new.params
m_new.summary()
#R sq value increases from 0.864 to 0.889
#pvalue of cc and Doors reduced significantly <0.05

#Final model with new data set
final_m=smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=corolla_new).fit()

#predictions for price with final model
price_pred = final_m.predict(corolla_new)

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(corolla_new.Price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(price_pred,final_m.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

np.mean(final_m.resid_pearson)
#mean of residuals is -1.9367549153512352e-12 === zero

########    Normality plot for residuals ######
# histogram
plt.hist(final_m.resid_pearson)

import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_m.resid_pearson, dist="norm", plot=pylab)

## Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
price_train,price_test  = train_test_split(corolla_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=price_train).fit()

# train_data prediction
train_pred = model_train.predict(price_train)

# train residual values  err/residual = Predicted value - Actual Value
train_resid  = train_pred - price_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(price_test)

# test residual values 
test_resid  = test_pred - price_test.Price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
