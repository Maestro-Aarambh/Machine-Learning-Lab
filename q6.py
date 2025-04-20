import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file = r"D:\Machine-Learning-Lab\Sensorless_drive_diagnosis.txt"
df = pd.read_csv(file, sep=' ', header=None)

X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


tree_counts = [20, 50, 100, 200, 500]
accuracies = []

for n in tree_counts:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Accuracy with {n} trees: {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(tree_counts, accuracies, marker='o', linestyle='-', color='blue')
plt.title("Random Forest Accuracy vs Number of Trees\n(Sensorless Drive Diagnosis Dataset)")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
