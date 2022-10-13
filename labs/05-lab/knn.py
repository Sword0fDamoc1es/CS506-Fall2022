class KNN:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_matrix = None
    
    def train(self):
        self.distance_matrix = ...

    def predict(self, example):
        return ...

    def get_error(self, predicted, actual):
        return sum(map(lambda x : 1 if (x[0] != x[1]) else 0, zip(predicted, actual))) / len(predicted)

    def test(self, test_input, labels):
        actual = labels
        predicted = (self.predict(test_input))
        print("error = ", self.get_error(predicted, actual))

# Add the dataset here

# Split the data 70:30 and predict.

# create a new object of class KNN

# plot a boxplot that is grouped by Species. 
# You may have to ignore the ID column

# predict the labels using KNN

# use the test function to compute the error

# from sklearn import datasets
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris


# iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target

# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# plt.figure(2, figsize=(8, 6))
# plt.clf()

# # Plot the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
# plt.xlabel("Sepal length")
# plt.ylabel("Sepal width")

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()