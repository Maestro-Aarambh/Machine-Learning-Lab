import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = np.array([2, 4, 5, 6, 8, 10, 12, 14, 16, 18]).reshape(-1, 1)
y = np.array([5, 9, 11, 13, 17, 21, 24, 28, 31, 36])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

m = LinearRegression()
m.fit(x_train, y_train)

y_pred = m.predict(x_test)

plt.scatter(x_train, y_train, color='blue', label='Train')
plt.plot(x_train, m.predict(x_train), color='red', label='Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
