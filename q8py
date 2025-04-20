import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

d = load_iris()
df = pd.DataFrame(data=d.data, columns=d.feature_names)
df['t'] = d.target

x = df[['sepal length (cm)']]
y = df['petal length (cm)']
y_bin = (y > 2).astype(int)

xtr, xts, ytr, yts = train_test_split(x, y_bin, test_size=0.3, random_state=42)

m = LinearRegression()
m.fit(xtr, ytr)

yp = m.predict(xts)
yp_bin = (yp > 0.5).astype(int)

cm = confusion_matrix(yts, yp_bin)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
