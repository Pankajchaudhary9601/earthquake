import pickle
import joblib
import sklearn


model_clone =joblib.load('earthquake_classification_model.pkl')

testing_data = [[  7.8  ,   4.   ,   7.   ,  20.   , 946.   , 379.   ,   0.   , 23.   ,  20.1  ,  -3.487, 100.082, 925.   , 830.   ,   0.   ,9.   ,   7.   , 439.   ,   1.   ]]


y_pred=model_clone.predict(testing_data)

print(y_pred)