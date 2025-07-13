import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import perceptron
import adaline

from sklearn.model_selection import train_test_split


URL       = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
DataFrame = pd.read_csv(URL, header=None, encoding='utf-8')

print(DataFrame.iloc()[:100])

y = np.where(DataFrame.iloc()[:100,4] == "Iris-setosa", -1, 1)
X = DataFrame.iloc()[:100,[0,2]].values

plt.scatter(X[:50,0],X[:50,1],marker='*')
plt.scatter(X[50:,0],X[50:,1],marker='x')
plt.legend(["Iris-setosa","Iris-versicolor"])
plt.show()

#Preprocessing step -- Spilt the data into training and test.
trainX, testX, trainY, testY = train_test_split(X,y,
                                                random_state=1,
                                                test_size=0.45,
                                                shuffle=True)

plt.scatter(trainX[:len(trainX),0],trainX[:len(trainX),1])
plt.scatter(testX[:len(testX),0],testX[:len(testX),1], marker='x')
plt.legend(["Training Set","Testing Set"])

plt.show()

#Train perceptron model
perceptronModel = perceptron.Perceptron(0.01,epoch=10)
perceptronModel.Train(trainX, trainY)
#Predict 
predictY = [perceptronModel.Predict(xi) for xi in testX ]
predictY = [yi - perceptronModel.Predict(xi)  for xi, yi in zip(testX,testY) ]
print("Perceptron Model: " , predictY)

plt.plot(range(1,len(perceptronModel.error)+1), perceptronModel.error,marker='o')
plt.show()

#Train adaline model
adalineModel = adaline.Adaline(0.0001,epoch=10)
adalineModel.Train(trainX, trainY)
predictY = [yi - adalineModel.Predict(xi)  for xi, yi in zip(testX,testY) ]
print("Adaline Model: " , predictY)

plt.plot(range(1,len(adalineModel.error)+1), adalineModel.error,marker='o')
plt.show()










