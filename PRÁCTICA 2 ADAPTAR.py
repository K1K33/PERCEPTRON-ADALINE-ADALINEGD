import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


#Perceptrón
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#clase Adaline
class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
#clase AdalineSGD
class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


# Cargar los datos
df = pd.read_csv('C:\\Users\\jerge\\Downloads\\familias-numerosas.csv', delimiter=',', encoding='latin1')


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Año']])
y = df['6-hijos']

# Mostrar los primeros registros del DataFrame
print(df.head())

#Gráfico para visualizar los datos de la provincia por año
plt.figure(figsize=(10, 6))
for provincia in df['Provincia'].unique():
    plt.plot(df[df['Provincia'] == provincia]['Año'], df[df['Provincia'] == provincia]['6-hijos'], label=provincia)
plt.xlabel('Año')
plt.ylabel('Número de familias con 6 hijos')
plt.title('Distribución de familias con 6 hijos por provincia y año')
plt.legend()
plt.show()

# Ajustar los datos para el gráfico de errores de misclasificación del Perceptrón
X_perceptron = df[['Año', '2-hijos']].values  
y_perceptron = df['6-hijos'].values


ppn = Perceptron()
ppn.fit(X_perceptron, y_perceptron)

# Gráfico de Errores de Misclasificación del Perceptrón
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron - Errors')
plt.show()

# Ajustar los datos para el gráfico de error de suma de cuadrados de Adaline con diferentes tasas de aprendizaje
X_adaline = X_scaled
y_adaline = y.values

# Lista de diferentes tasas de aprendizaje
etas = [0.01, 0.0001, 0.00001]

#Gráfico de error de suma de cuadrados de Adaline con diferentes tasas de aprendizaje
plt.figure(figsize=(10,6))
for eta in etas:
    adaline = AdalineGD(eta=eta, n_iter=50)
    adaline.fit(X_adaline, y_adaline)
    plt.plot(range(1, len(adaline.cost_) + 1), adaline.cost_, label=f'eta = {eta}')

plt.xlabel('Epochs')
plt.ylabel('Sum Squared Error')
plt.title('Adaline - Learning rate')
plt.legend()
plt.show()

# Función para trazar regiones de decisión
def plot_decision_regions_adaline(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    label=cl,
                    edgecolors='black')


X_adaline = df[['Año', '2-hijos']].values  
y_adaline = df['6-hijos'].values


adaline = AdalineGD(eta=0.0001, n_iter=200)
adaline.fit(X_adaline, y_adaline)

# Graficar las regiones de decisión de Adaline
plt.figure(figsize=(10, 6))
plot_decision_regions_adaline(X_adaline, y_adaline, classifier=adaline)
plt.xlabel('Año')
plt.ylabel('Número de familias con 6 hijos')
plt.title('Regiones de decisión de Adaline')
plt.legend(loc='upper left')
plt.show()


X_adaline_normalized = scaler.fit_transform(df[['Año', '2-hijos']])
y_adaline_normalized = df['6-hijos'].values

# Lista de diferentes tasas de aprendizaje
etas = [0.01, 0.0001, 0.00001]

#Gráfico de error de suma de cuadrados de Adaline con diferentes tasas de aprendizaje después de la normalización
plt.figure(figsize=(10,6))
for eta in etas:
    adaline_normalized = AdalineGD(eta=eta, n_iter=50)
    adaline_normalized.fit(X_adaline_normalized, y_adaline_normalized)
    plt.plot(range(1, len(adaline_normalized.cost_) + 1), adaline_normalized.cost_, label=f'eta = {eta}')

plt.xlabel('Epochs')
plt.ylabel('Sum Squared Error')
plt.title('Adaline - Learning rate after feature scaling')
plt.legend()
plt.show()

# Ajustar los datos para el gráfico de costo promedio de Adaline utilizando el Descenso de Gradiente Estocástico
X_adaline_sgd = scaler.fit_transform(df[['Año', '2-hijos']])
y_adaline_sgd = df['6-hijos'].values

#Gráfico de costo promedio de Adaline con SGD
adaline_sgd = AdalineSGD(eta=0.01, n_iter=50, random_state=1)
adaline_sgd.fit(X_adaline_sgd, y_adaline_sgd)

plt.plot(range(1, len(adaline_sgd.cost_) + 1), adaline_sgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('Adaline with Stochastic Gradient Descent')
plt.show()

# Función para trazar regiones de decisión del Perceptrón
def plot_decision_regions_perceptron(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    label=cl,
                    edgecolors='black')

# Ajuste de los datos para el gráfico de regiones de decisión del Perceptrón
X_perceptron_decision = df[['Año', '2-hijos']].values  
y_perceptron_decision = df['6-hijos'].values


ppn_decision = Perceptron(eta=0.1, n_iter=10)
ppn_decision.fit(X_perceptron_decision, y_perceptron_decision)

# Gráfico de regiones de decisión del Perceptrón
plt.figure(figsize=(10, 6))
plot_decision_regions_perceptron(X_perceptron_decision, y_perceptron_decision, classifier=ppn_decision)
plt.xlabel('Año')
plt.ylabel('Número de familias con 6 hijos')
plt.title('Regiones de decisión del Perceptrón')
plt.legend(loc='upper left')
plt.show()



