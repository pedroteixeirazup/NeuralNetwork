from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target

redeNeural = MLPClassifier(verbose=true,
                            max_iter=1000,
                            tol=0.00001,
                            activation='logistic',
                            learning_rate_init=0.3)
redeNeural.fit(entradas,saidas)