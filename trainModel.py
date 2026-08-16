#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import joblib


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef


# In[10]:


# "Wine Quality" Dataset

url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

df_red = pd.read_csv(url_red, sep=';')
df_white = pd.read_csv(url_white, sep=';')

print("Dataset loading completed")


# In[20]:


df_red['wine_type'] = 1   # 1 for Red
df_white['wine_type'] = 0 # 0 for White
df = pd.concat([df_red, df_white], ignore_index=True)

df['quality_label'] = (df['quality'] > 5).astype(int)

X = df.drop(['quality', 'quality_label'], axis=1) 
y = df['quality_label']        

print(f"Dataset Shape: {X.shape}")


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)
print("Saved test_data.csv")


# In[26]:


os.makedirs('model', exist_ok=True)

models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42)
}


# In[28]:


evaluation_metrics = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    evaluation_metrics.append({
        'ML Model Name': name, 'Accuracy': round(acc, 4), 'AUC': round(auc, 4),
        'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4), 'MCC': round(mcc, 4)
    })
    
    file_name = name.replace(' ', '_')
    joblib.dump(model, f'model/{file_name}.pkl')


# In[30]:


results_df = pd.DataFrame(evaluation_metrics)
print(results_df.to_markdown(index=False))


# In[ ]:




