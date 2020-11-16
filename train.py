import pickle

import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

print("Downloading data...")
data = pd.read_csv("https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv")
x = data[["temperature", "humidity", "windspeed"]]
y = data["count"]

X_train, X_test, y_train,y_test = train_test_split(x,y, 
                                                   test_size=0.1,
                                                   random_state=1)

print("Training model...")                                                   
model = xgb.sklearn.XGBClassifier(nthread=1, seed =1)
model.fit(X_train, y_train)


preds = model.predict(X_test)
score = r2_score(y_test, preds)
print(f"R2 score on test-set is {score}")

print("Saving model...")
with open("./model/xgbmodel.pkl", "wb") as f:
    pickle.dump(model,f)
print("as ./model/xgbmodel.pkl")
