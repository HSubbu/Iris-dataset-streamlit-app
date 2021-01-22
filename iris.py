import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np 

df = pd.read_csv('data/iris.csv')
df.drop('Id',axis=1,inplace=True)
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

X = df.drop('Species',axis=1)
y = df['Species']

clf = LogisticRegression()
clf.fit(X,y)

X_test = np.array([[5.1,3.5,1.4,0.2]])
  
# Save the model as a pickle in a file 
joblib.dump(clf, 'iris.pkl') 
  
# Load the model from the file 
logreg_from_joblib = joblib.load('iris.pkl')  
  
# Use the loaded model to make predictions 
print(logreg_from_joblib.predict(X_test)) 
