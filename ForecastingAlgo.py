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


# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)

# Y Prediction
Y_pred = reg.predict(X)


# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel(result[0])
plt.ylabel(result[1])
plt.legend()
plt.show()

#!!!!!!!!!!!!!! end visual plotting section!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!