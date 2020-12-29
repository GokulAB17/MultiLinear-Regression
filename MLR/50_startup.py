#Prepare a prediction model for profit of 50_startups data.
#Do transformations for getting better predictions of profit and
#make a table containing R^2 value for each prepared model.

#R&D Spend -- Research and devolop spend in the past few years
#Administration -- spend on administration in the past few years
#Marketing Spend -- spend on Marketing in the past few years
#State -- states from which data is collected
#Profit  -- profit of each state in the past few years

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
startup = pd.read_csv(r"C:\Users\Administrator.Gokulbhasi\Desktop\DataScience\Assignments\MLR\50_startups.csv")
startup

#Data preprocessing
#Mapping values for col "State" to dummy variables for model building 
new_startup=startup
new_startup.iloc[:,3]=startup.State.map({"New York":1,"California":2,"Florida":3})

#checking record of first 20
new_startup.head(20)

#column names change for ease of operation
new_startup.columns="RnD","Admin","MarkSpnd","State","Profit"

#Exploratory data analysis
new_startup.corr()
#we see collinearity between MarkSpnd & RnD

#plotting scatter plot between all combinations in data set
import seaborn as sns
sns.pairplot(new_startup)

#Modelbuiding
import statsmodels.formula.api as smf

model1 = smf.ols('Profit~RnD+Admin+MarkSpnd+State',data=new_startup).fit()

model1.params

model1.summary()
#Rsquared value=0.951
#We see Admin,Markspnd,State have p values > 0.05

#preparing model with Markspnd only
m2=smf.ols("Profit~MarkSpnd",data=new_startup).fit()

m2.summary()
#pvalue <0.05

#Model with Admin only
m3=smf.ols("Profit~Admin",data=new_startup).fit()

m3.summary()
#pvlaue>0.05

#Model with State only
m4=smf.ols("Profit~State",data=new_startup).fit()
m4.summary()

#pvlalue>0.05
#pvalue of Admin and State is insignificant

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 48,46,49 are showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals
#Dropping records 48,46,49
new_startup1=new_startup.drop(new_startup.index[[48,46,49]],axis=0)

#new model 
model2 = smf.ols('Profit~RnD+Admin+MarkSpnd+State',data=new_startup1).fit()
model2.summary()
#Rsquare value improved to 0.961

# calculating VIF's values of independent variables
rsq_RnD = smf.ols('RnD~Admin+MarkSpnd+State',data=new_startup1).fit().rsquared  
vif_RnD = 1/(1-rsq_RnD) #2.708


rsq_Admin = smf.ols('Admin~RnD+MarkSpnd+State',data=new_startup1).fit().rsquared  
vif_Admin = 1/(1-rsq_Admin) #1.233

rsq_MarkSpnd = smf.ols('MarkSpnd~RnD+Admin+State',data=new_startup1).fit().rsquared  
vif_MarkSpnd = 1/(1-rsq_MarkSpnd) #2.692

rsq_State = smf.ols('State~RnD+MarkSpnd+Admin',data=new_startup1).fit().rsquared  
vif_State = 1/(1-rsq_State) #1.00
#no Vif values of variables is >10 so no variables to be removed from predictionmodel

# Added varible plot 
sm.graphics.plot_partregress_grid(model2)

#final model is model2
finalmodel=model2
finalmodel.summary()
#Rsquare value is 0.961 which is increased from 0.951 by removing 3 records

profit_predict=finalmodel.predict(new_startup1)

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(new_startup1.Profit,profit_predict,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
profit_predict.corr(new_startup1.Profit)# corr =0.980

# Residuals VS Fitted Values 
plt.scatter(profit_predict,finalmodel.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

#Normality
plt.hist(finalmodel.resid_pearson)


np.var(finalmodel.resid_pearson)

import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(finalmodel.resid_pearson, dist="norm", plot=pylab)

### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(new_startup1,test_size = 0.2) # 50% size

# preparing the model on train data 

model_train = smf.ols("Profit~RnD+Admin+MarkSpnd+State",data=new_startup1).fit()

# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values  err/residual = Predicted value - Actual Value
train_resid  = train_pred - startup_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid  = test_pred - startup_test.Profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

model_log = smf.ols('Profit~np.log(RnD+Admin+MarkSpnd+State)',data=new_startup1).fit()
model_log.summary()#Rsq =0.726

model_exp=smf.ols("np.log(Profit)~RnD+Admin+MarkSpnd+State",data=new_startup1).fit()
model_exp.summary() #Rsq=0.931

model_quad=smf.ols("Profit~RnD*RnD+Admin*Admin+MarkSpnd*MarkSpnd+State*State",data=new_startup1).fit()
model_quad.summary() #Rsq=0.961

Table_Rsq=pd.DataFrame(columns=["Transformations","Rsq"])
x1=["normal","logaritmic","exponential","quadratic"]
x2=[0.961,0.726,0.931,0.961]
Table_Rsq.Transformations=pd.Series(x1)
Table_Rsq.Rsq=pd.Series(x2)
Table_Rsq

#table for all Rsquare values for diff transformations
