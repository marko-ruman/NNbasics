import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# NN model definition

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


# DATA

x_data = torch.Tensor([[10.0], [9.0], [3.0], [2.0]])
y_data = torch.Tensor([[90.0], [80.0], [50.0], [30.0]])

# NN PART

model_nn = MLP()

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model_nn.parameters(), lr=0.001)


for epoch in range(50000):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model_nn(x_data)

    # Compute Loss
    loss = criterion(y_pred, y_data)

    # Backward pass
    loss.backward()

    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

y_pred_nn = model_nn(x_data)

# Lin Reg part

model_lin = LinearRegression()

model_lin.fit(x_data, y_data)

y_pred_lin = model_lin.predict(x_data)


# Drawing part

with torch.no_grad():
    plt.clf()
    plt.plot(x_data, y_data, 'go', label='True data', alpha=0.5)
    plt.plot(x_data, y_pred_nn, '--', label='Predictions NN', alpha=0.5)
    plt.plot(x_data, y_pred_lin, '--', label='Predictions Lin Reg', alpha=0.5)
    plt.legend(loc='best')
    plt.show()