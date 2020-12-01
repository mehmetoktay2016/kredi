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





#df = pd.read_csv("data/credit_risk.csv", index_col = 0)
df = pd.read_csv("/Users/user/PycharmProjects/pythonProject5/12_hafta/kredi_riski/data/credit_risk.csv", index_col = 0)

df = df.drop(["Sex","Job","Housing","Saving accounts","Checking account","Purpose"] , axis = 1)
df.loc[df.Risk == "good" , "Risk"] = 1
df.loc[df.Risk == "bad" , "Risk"] = 0
df.Risk = df.Risk.astype("category")

df.head()


X = df.drop("Risk", axis=1)
y = df[["Risk"]]




from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, y)

y_pred = lr.predict(X)


pickle.dump(lr, open('lrrr_model.pkl','wb'))

print("Model Kaydedildi")



