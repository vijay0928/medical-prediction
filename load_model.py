import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
train_data=pd.read_csv(r"C:\Users\digital\PycharmProjects vft\data_frame\train_scaled.csv")
X = train_data.drop(['opted_encoded'], axis=1)
y = train_data['opted_encoded']
print(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_predictions = loaded_model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print(f"Accuracy on validation set: {accuracy}")
print(X_val.iloc[0])