import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

f = r"D:\Machine-Learning-Lab\Sensorless_drive_diagnosis.txt"
df = pd.read_csv(f, sep=r'\s+', header=None)

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=42)

k_list = ['linear', 'poly', 'rbf']
acc_list = []

for k in k_list:
    m = SVC(kernel=k)
    m.fit(x_tr, y_tr)
    y_p = m.predict(x_te)
    acc = accuracy_score(y_te, y_p)
    acc_list.append(acc)
    print(f"Accuracy with {k} kernel: {acc:.4f}")

# Simple line plot
plt.figure(figsize=(8, 5))
plt.plot(k_list, acc_list, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title("SVM Accuracy with Kernels")
plt.xlabel("Kernel")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()
