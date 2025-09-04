# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset, define features (X) and target (y), then split into train and test sets.
2. Initialize and train a Linear Regression model with training data.
3. Predict test data, evaluate with MAE and R², and make custom predictions (e.g., 6.1 hrs).
4. Plot scatter of data and regression line with proper labels and title.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Elfreeda Jesusha J
RegisterNumber:  212224040084
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
data=pd.read_csv("student_scores.csv")
data
X=data[['Hours']]
y=data[['Scores']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
print("R2 score:",r2_score(y_test,y_pred))
pred =model.predict([[6.1]])
pred
plt.scatter(X,y,color='blue')
plt.plot(X,model.predict(X),color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression-Student Marks Prediction")
plt.plot()

```

## Output:
<img width="435" height="338" alt="exp2ml" src="https://github.com/user-attachments/assets/1e5de2b4-0550-479d-97c6-c942879f0d61" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
