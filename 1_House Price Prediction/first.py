from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


import numpy as np
import pandas as pd

dataset = fetch_california_housing()
df = pd.DataFrame(dataset["data"],columns=dataset["feature_names"]) # type: ignore

X = df
y = dataset["target"] # type: ignore

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
model = LinearRegression()
model.fit(X_train,y_train)
# print(model.predict(X_test))
Eror_MAE = mean_absolute_error(y_test,model.predict(X_test))
# print(Eror_MAE)

meany_train = np.mean(y_train)
y_baseline = np.full_like(y_test,fill_value=meany_train)
mae_baseline = mean_absolute_error(y_test,y_baseline)
# print(mae_baseline)

model_tree = DecisionTreeRegressor()
model_tree.fit(X_train,y_train)
print(model_tree.predict(X_test))
Eror_test = mean_absolute_error(y_test,model_tree.predict(X_test))
Eror_train = mean_absolute_error(y_train,model_tree.predict(X_train))
print("Error test :",Eror_test)
print("Error train :",round(Eror_train,4))
print(y_test)


list1 = []
list2 = []

for i in [3,5,10,20]:
    model_tree = DecisionTreeRegressor(max_depth=i)
    model_tree.fit(X_train,y_train)
    Eror_test = mean_absolute_error(y_test,model_tree.predict(X_test))
    Eror_train = mean_absolute_error(y_train,model_tree.predict(X_train))

    list1.append(Eror_train)
    list2.append(Eror_test)
d = {"Max depth":[3,5,10,20],"Train MAE" : list1, "Test MAE" :list2}



Test_mae_table = pd.DataFrame(data=d)
print(Test_mae_table)