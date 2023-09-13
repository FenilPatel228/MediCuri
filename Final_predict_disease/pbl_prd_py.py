import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('.Training.csv')
data.head()
X = data.drop('prognosis', axis=1)
y = data['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgboost_model = XGBClassifier(n_estimators=100, objective='multi:softprob', random_state=42)
bayes = GaussianNB()
# tree = DecisionTreeClassifier(random_state=42)
# svc = SVC(random_state=42)
random_forest_model.fit(X_train, y_train_encoded)
xgboost_model.fit(X_train, y_train_encoded)
bayes.fit(X_train, y_train_encoded)
# tree.fit(X_train, y_train_encoded)
# svc.fit(X_train, y_train_encoded)
rf_prd = random_forest_model.predict(X_test)
xgboost_prd = xgboost_model.predict(X_test)
bayes_prd = bayes.predict(X_test)
# tree_prd = tree.predict(X_test)
# svc_prd = svc.predict(X_test)
rf_acc = accuracy_score(y_test_encoded, rf_prd)
xgboost_acc = accuracy_score(y_test_encoded, xgboost_prd)
bayes_acc = accuracy_score(y_test_encoded, bayes_prd)
# tree_acc = accuracy_score(y_test_encoded, tree_prd)
# svc_acc = accuracy_score(y_test_encoded, svc_prd)
print(rf_acc,xgboost_acc,bayes_acc)
def predict_disease(symptoms):
    # Convert the input symptom names into a binary format using the column names
    input_data = pd.DataFrame([[1 if col in symptoms else 0 for col in X.columns]], columns=X.columns)
    
    # Make predictions using individual models
    rf_prediction = random_forest_model.predict(input_data)
    xgboost_prediction = xgboost_model.predict(input_data)
    bayes_prediction = bayes.predict(input_data)
    # tree_prediction = tree.predict(input_data)
    # svc_prediction = svc.predict(input_data)
    
    # Decode the predicted label using the label encoder
    # predicted_disease = np.array([le.inverse_transform([rf_prediction]),le.inverse_transform([xgboost_prediction]),le.inverse_transform([bayes_prediction]),le.inverse_transform([tree_prediction]),le.inverse_transform([svc_prediction])])
    predicted_disease = np.array([le.inverse_transform([rf_prediction]),le.inverse_transform([xgboost_prediction]),le.inverse_transform([bayes_prediction])])
    predicted_disease = np.unique(predicted_disease)
    
    # res = f"""
    # rf_prediction = {le.inverse_transform([rf_prediction])}
    # xgboost_prediction = {le.inverse_transform([xgboost_prediction])}
    # bayes_prediction = {le.inverse_transform([bayes_prediction])}
    # """
    
    # print(res)
    
    return (predicted_disease)

# Example usage
input_symptoms = ["high_fever", "nausea", "pain_behind_the_eyes", 'headache']
predicted_disease = predict_disease(input_symptoms)
print("Predicted Disease:", predicted_disease)
input_symptoms = ["fatigue", "restslessness", "diarrhoea", 'vomiting']
predicted_disease = predict_disease(input_symptoms)
print("Predicted Disease:", predicted_disease)
input_symptoms = ["fatigue", "itching", "headache", 'vomiting', 'high_fever']
predicted_disease = predict_disease(input_symptoms)
print("Predicted Disease:", predicted_disease)
input_symptoms = ["nausea", "itching", "headache", 'vomiting', 'high_fever']
predicted_disease = predict_disease(input_symptoms)
print("Predicted Disease:", predicted_disease)
input_symptoms = ["fatigue", "cough", "headache", 'chills', 'high_fever', 'dizziness']
predicted_disease = predict_disease(input_symptoms)
print("Predicted Disease:", predicted_disease)
input_symptoms = ["high_fever"]
predicted_disease = predict_disease(input_symptoms)
print("Predicted Disease:", predicted_disease)
# import pickle

# with open('rf_dis_model.pkl', 'wb') as model_rf:
#     pickle.dump(random_forest_model, model_rf)
    
# with open('xg_dis_model.pkl', 'wb') as model_xg:
#     pickle.dump(xgboost_model, model_xg)

# with open('bayes_dis_model.pkl', 'wb') as model_bayes:
#     pickle.dump(bayes, model_bayes)
