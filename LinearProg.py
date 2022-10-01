#George Curry
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#create data set
dataSet =  pd.DataFrame({
    'Months': [1,2,3,4,5,6,7,8,9,10,11],
    'Price': [10.25, 10.45, 10.87, 10.75, 11.2, 12.01, 12.25, 12.95, 12.80, 13.05, 14.2]
})
print(dataSet)

#data has linear progression, so lets use the OLS()Function

#response variable
y = dataSet['Price']

#independent variable
x = dataSet['Months']

#add a constant to predictor
x = sm.add_constant(x)

#fit linear regression
model = sm.OLS(y,x).fit()
predict = model.predict(x)
#View
print(model.summary())

#The fitted regression line is Price(y) = 9.63+(0.38*(months))
result = "{:.2f}".format(9.63+(0.38*12))
print("Estmated Month 12 price: "+str(result))

#show regression data on graph
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model,'Months',fig=fig)
plt.show()

#Based on the data, I would buy the asset because the trend is positive.