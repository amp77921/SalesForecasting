# Importing Necessary Libraries
# to install modules : C:\Users\houjreed\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.9>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (10.0, 5.0)

# User Input
print("Enter name of CSV file")
csv_file_name = input()
csv_file_name = csv_file_name.upper()
csv_file_name = csv_file_name + ".CSV"

# Reading Data
data = pd.read_csv(csv_file_name)
result = data.columns
ColumnA = result[0]
ColumnB = result[1]
print("Column A = " + ColumnA)
print("Column B = " + ColumnB)

# print(data.shape)
data.head()

# Collecting X and Y ; X = independent var, Y = prediction
X = data[ColumnA].values  # names of independent var header of data in csv file
Y = data[ColumnB].values # names of dependent var header of data in csv file

# Mean X and Y 
my_df = pd.DataFrame(data)
XstdDev = my_df[ColumnA].std()
YstdDev = my_df[ColumnB].std()
mean_x = my_df[ColumnA].mean()
mean_y = my_df[ColumnB].mean()
min_x = my_df[ColumnA].min()
max_x = my_df[ColumnA].max()

# Total number of values
totalNumValues = len(X)             #(degrees of freedom =  ((totalNumValues - number of independent variables "X in this equation") -1)
df = ((totalNumValues - 1) -1)      #df = ((totalNumValues - 1)-1)------------there's only one X variable in equation; want df to be high
mDF = 1                             # number of variables (X) this model is testing

# Using the formula to calculate b1 and b2
#SSR = difference from mean to expected value (along slope (m)) (sum of squares due to regression)
#SSE = difference from expected value to observed value (also known as unaccounted for error) (sum of squares due to error)
#SST = sum of the difference from mean to expected value and diff from expected to observed value (total deviation from mean)
#r2 = SSR/SST      high r2 means regression line is really good fitting (low SSE); low r2 means not so good fitting (high sse) so you want r2 to be high.  High = more accurate prediction. values from 0 to 1

numer = 0
denom = 0
for i in range(totalNumValues):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)          #SSR
    denom += (X[i] - mean_x) ** 2                       #SST = SSR + SSE
b1 = numer / denom                                      #scale factor (coefficient), b1=r2
                                                        #standard deviation of sample set = XstdDev

sError = XstdDev / np.sqrt(totalNumValues ** 2)         #standard error 
b0 = mean_y - (b1 * mean_x)                             #bias coefficient    

if (b1 >= 0.5):
    isPos = "Strong positive correlation"
elif (b1 > 0.0 and b1 < 0.5):
    isPos = "Weak positive correlation"
elif (b1 <= 0.0):
    isPos = "No correlation"

my_df = pd.DataFrame(data)
XstdDevstr = str(XstdDev)
YstdDevstr = str(YstdDev)

#print(data[result[0]].describe())
#print('')
#print(data[result[1]].describe())

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((totalNumValues, 1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)

# Y Prediction
Y_pred = reg.predict(X)

# Calculating RMSE and R2 Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)
    
r2Percent = r2_score * 100

mSS = mse * r2_score
rSS = mse - mSS
mMeanSqr = mSS / mDF
rMeanSqr = rSS / (df - 1)
tMeanSqr = mse / df
fValue = rMeanSqr / tMeanSqr

#calculate adjusted r2 values (use when using more than 1 X variable - else will equal R2)
adjustR2Numerator = (1-r2_score) * (totalNumValues - 1)
adjustR2Demonator = totalNumValues - mDF - 1
adjustR2 = 1 - (adjustR2Numerator / adjustR2Demonator)

# calculate t-score
# use  totalNumValues , alpha <0.05 (5% confidence level), 2 tail
# 2 tail
alphaValue = 0.025
tScore2Tail = stats.t.ppf(1-(alphaValue/2), df)

# 1 tail (left tail)
alphaValue1 = 0.05
tScore1Tail = stats.t.ppf(1-alphaValue1, df)

t1 = b1 / sError

# P value
pValue = stats.t.sf(abs(tScore2Tail), df)

confInterval = (1.96 * (XstdDev / (np.sqrt(totalNumValues))))
lowerCI = mean_x - confInterval
upperCI = mean_x + confInterval

# Predict by input value
Y_pred = reg.predict(X)
if (b1 > 0.0):
  print('Enter independent variable number to test: ')
  NewValue = input()
  NewValue = float(NewValue)
  NewY = (b0 + (b1 * NewValue))
elif (b1 <= 0.0):
  print("Negative Corelation, cannot predict future values with confidence")


# print(reg.summary())

# OUTPUT
print("Scale factor coefficient B1: " + str(b1))
print("Bias coefficient B0: " + str(b0))
print(" ")
print('Column A std dev: ' + XstdDevstr)
print('Column B std dev: ' + YstdDevstr)
print(" ")
print("Model describes " + str(round(r2Percent,2)) + "% " + "chance of variation in dependent variable is explained by the independent variable")
print(" ")
print("SOURCE   " + "SUM OF SQUARES     " + " DEGREES OF FREEDOM    " + " Mean Square    ")
print("Model    " + str(round(mSS,2)) + "         " + "       " + str(mDF) + "                       " + str(round(mMeanSqr,2)))
print("Residual " + str(round(rSS,2)) + "       " + "      " + str(df-1) + "                    " + str(round(rMeanSqr,2)))
print("Total    " + str(round(mse,2)) + "       " + "      " + str(df) + "                    " + str(round(tMeanSqr,2)))
print(" ")
print("Total number of observations tested: " + str(totalNumValues))
print("F value (" + str(mDF) + ", " + str(df - 1) +") : " + str(round(fValue,2)))
print("R-squared: " + str(round(r2_score,2)))
print("Adjusted R-squared: " + str(round(adjustR2,2)))
print("Root Mean Square Error: " + str(round(rmse,2)))

if (pValue <= alphaValue):
    randomChance = "NOT RANDOM"
elif (pValue > alphaValue):
    randomChance = "RANDOM"  
print("") 
print("") 
print("CONCLUSION: 95% confidence that correlation is " + randomChance)
#print(" ")
#print("        " + "Coeff   " + "Std. Error     " + " t value    " + " P>|t|    " + "95% confidence level")
#print("Sample: " + str(round(b1,2)) + "      " + str(round(sError,2)) + "          " + str(round(t1,2)) + "        " + str(round(pValue,2)) + "      " + str(round(lowerCI,2))+ "      " + str(round(upperCI,2)))

if (b1 > 0.0):
  print("") 
  print("CONCLUSION: " + isPos.upper()) 
  print("")
  print("Value inputed: " + str(NewValue))
  print("Predicted value: " + str(NewY))
  print("95% confidence predicted range is between: " + str(NewY - (XstdDev * 2)) + " and " + str(NewY + (XstdDev * 2)))

#!!!!!!!!!!!!!! begin visual plotting section!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel(result[0])
plt.ylabel(result[1])
plt.legend()
plt.show()

#!!!!!!!!!!!!!! end visual plotting section!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!