# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:44:14 2024

@author: Yamaç
"""

from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
import sklearn

import pandas
import numpy as np
import torch

  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  



#Data preprocessing y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ct = ColumnTransformer(transformers=[('encoder', sklearn.preprocessing.OneHotEncoder(), [0])], remainder='drop')
y_train = ct.fit_transform(y_train)
y_test = ct.transform(y_test)
y_train = pandas.DataFrame(y_train).drop(0, axis=1)
y_test = pandas.DataFrame(y_test).drop(0, axis=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values.astype(np.float32), dtype=torch.float32)
y_test = torch.tensor(y_test.values.astype(np.float32), dtype=torch.float32)

# Model architecture
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(X_train.shape[1], 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instantiate the model
model = Model()

# Loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
with torch.no_grad():
    model.eval()
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5).float()  # Convert to binary predictions
    test_accuracy = torch.sum(y_pred == y_test) / y_test.shape[0]
    print(f'Test Accuracy: {test_accuracy.item():.4f}')
    
#Testing the model
y_pred = model(X_test)   

y_pred = y_pred.detach().numpy()

#Making y_pred binary
for i in range(len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
    
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='macro')  # Macro-averaging
print("F1-score (Macro):", f1)