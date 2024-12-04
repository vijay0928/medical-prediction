import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


train_data=pd.read_csv(r"C:\Users\digital\PycharmProjects vft\data_frame\train_scaled.csv")
X = train_data.drop(['opted_encoded'], axis=1)
y = train_data['opted_encoded']
print(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

val_predictions = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print(f"Accuracy on validation set: {accuracy}")
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)