import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

accident_data = pd.read_csv("accident_data.csv")

selected_columns = accident_data[accident_data['Severity'].isin(["Grievous Injury", "Fatal", "Damage Only", "Simple Injury"])]
selected_columns = selected_columns[["Road_Type", "Weather", "Accident_Spot", "Accident_Location", "Accident_SubLocation", "Collision_Type", "Severity"]]

selected_columns = selected_columns.dropna()

# Define the range of learning rates


learning_rates = [i / 10 for i in range(1, 11)]

#############################################################
#############################################################
#############################################################


accuracies = []
execution_times = []

# Iterate over each learning rate
for learning_rate in learning_rates:
    X = selected_columns.drop(columns=["Severity"])
    y = selected_columns["Severity"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = CatBoostClassifier(iterations=20, learning_rate=learning_rate, depth=6, random_state=42)
    
    start_time = time.time()
    
    clf.fit(X_train, y_train, cat_features=["Road_Type", "Weather", "Accident_Spot", "Accident_Location", "Accident_SubLocation", "Collision_Type"], verbose=False)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    


    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(learning_rates, accuracies, marker='o', color='blue')
plt.title('Learning Rate vs Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot execution time
plt.subplot(2, 1, 2)
plt.plot(learning_rates, execution_times, marker='x', color='red')
plt.title('Learning Rate vs Execution Time')
plt.xlabel('Learning Rate')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)

plt.tight_layout()
plt.show()

