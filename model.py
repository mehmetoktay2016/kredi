#Kütüphanelerin yuklenmesi
import pickle #python nesnelerini kaydetmek ve cagirmak icin kullanilir.
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score





df = pd.read_csv("data/churn.csv", index_col = 0)
#df = pd.read_csv("/Users/user/PycharmProjects/pythonProject5/12_hafta/kredi/churn_deployment/data/churn.csv", index_col = 0)
pd.pandas.set_option("display.max_columns", None)
df.head()
df = df.drop(["CustomerId","Surname","Geography","NumOfProducts"] , axis = 1)

df.loc[df.Gender == "Female" , "Gender"] = 1
df.loc[df.Gender == "Male" , "Gender"] = 0
df.Gender = df.HasCrCard.astype("int")

df.head()





X = df.drop("Exited", axis=1)
y = df[["Exited"]]




from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, y)

y_pred = lr.predict(X)


pickle.dump(lr, open('lrrs_model.pkl','wb'))

print("Model Kaydedildi")



