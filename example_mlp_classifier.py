import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# Plotting function

def plot_regions(x_data, y_data, model, model_type="torch"):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = x_data[:, 0].min() - 0.5, x_data[:, 0].max() + 0.5
    y_min, y_max = x_data[:, 1].min() - 0.5, x_data[:, 1].max() + 0.5
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    if model_type == "logreg":
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    elif model_type == "torch":
        Z = torch.max(model(torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()), dim=1)[1].detach().numpy()

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, edgecolors="k", cmap=plt.cm.Paired)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()


# NN model definition

class BigMLPClassifier(torch.nn.Module):
    def __init__(self):
        super(BigMLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 3)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


# import some data to play with

iris = datasets.load_iris()
x_data = iris.data[:, :2]  # we only take the first two features.
y_data = iris.target


mlp_classifier = MLPClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(mlp_classifier.parameters(), lr=0.01)

for epoch in range(50000):
    optimizer.zero_grad()
    # Forward pass
    y_pred = mlp_classifier(torch.tensor(x_data).float())

    # Compute Loss
    loss = criterion(y_pred, torch.tensor(y_data).long())

    # Backward pass
    loss.backward()

    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))



# Create an instance of Logistic Regression Classifier and fit the data.
logreg = LogisticRegression(C=1e5)
logreg.fit(x_data, y_data)

plot_regions(x_data, y_data, logreg, "logreg")
plot_regions(x_data, y_data, mlp_classifier, "torch")
