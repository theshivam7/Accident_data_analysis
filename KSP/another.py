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
X = selected_columns.drop(columns=["Severity"])
y = selected_columns["Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, random_state=42)

start_time = time.time()

clf.fit(X_train, y_train, cat_features=["Road_Type", "Weather", "Accident_Spot", "Accident_Location", "Accident_SubLocation", "Collision_Type"],verbose=False)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()

execution_time = end_time - start_time

y_pred_flat = [item[0] for item in y_pred]


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

end_time = time.time()
execution_time = end_time - start_time
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test[:50])), y_test[:50], marker='o', linestyle='-', color='blue', label='Actual Severity')
plt.plot(range(len(y_pred_flat[:50])), y_pred_flat[:50], marker='x', linestyle='--', color='red', label='Predicted Severity')
plt.title('Actual vs Predicted Severity (Subset of Testing Data)')
plt.xlabel('Data Point')
plt.ylabel('Severity')
plt.legend()
plt.grid(True)


plt.text(0.45, 0.90, f'Learning Rate: {0.1}\nIterations: {200}\nAccuracy: {accuracy:.2f}\nExecution Time: {execution_time:.2f} seconds',
         horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()