import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
from perceptron import Perceptron
#Setting figure size
rcParams["figure.figsize"] = 10, 5

dataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

print(dataFrame.iloc[145:150, 0:5])

# Y = Class Labels (Iris-Setosa and Iris-Versicolor)
y = dataFrame.iloc[0:100, 4].values
print(y)

# Se Iris-Setosa = -1 Se Iris-Versicolor = 1
y = np.where(y == 'Iris-setosa', -1, 1)
print(y)

# Matriz com características 0 e 2 
X = dataFrame.iloc[0:100, [0, 2]].values
print(X)


plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='VersiColor')
plt.xlabel('Tamanho Pétala')
plt.ylabel('Tamanho Sépala')
plt.legend(loc='upper left')
plt.show()

pn = Perceptron(0.1, 10)
pn.fit(X, y)

plt.plot(range(1, len(pn.errors_) + 1), pn.errors_, marker='o')
plt.xlabel('Iterações')
plt.ylabel('Erros de Classificações')
plt.show()