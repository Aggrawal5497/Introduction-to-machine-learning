import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

x = np.linspace(0, 100, 100)
delta = np.random.uniform(-5, 5, x.size)
y = 0.6*x + 3 + delta

plt.scatter(x, y)
plt.title("Data Set")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

def MSELoss(predicted, target):
    dif = (predicted - target) ** 2
    return np.mean(dif)

def GradientDecent(theta, x, predicted, target, learning_rate = 2e-05):
    dif = predicted - target
    difprod = np.multiply(dif, x)
    gradient = np.mean(difprod, axis = 0)
    change = gradient.reshape(-1, 1) * learning_rate
    return theta - change


x_val = np.ones(shape=(len(x), 2))
y_val = np.array(y).reshape(-1, 1)
x_val[:, 1] = np.array(x).T

weights = np.random.rand(2, 1)

fig = plt.gcf()
fig.set_size_inches(10, 5.5)
fig.show()
fig.canvas.draw()
fig.canvas.set_window_title('Linear Regression')

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


loss = []

for j in range(1, 101):
    pred = np.matmul(x_val, weights)
    l = MSELoss(pred, y_val)
    loss.append(l)
    weights = GradientDecent(weights, x_val, pred, y_val)
    ax1.clear()
    ax2.clear()

    ax1.set_title("Linear Regression Learning\n Iteration {}".format(j))
    ax1.set_xlabel("Input")
    ax1.set_ylabel("Output")

    ax2.set_title("Loss Graph\n Current Loss : {:.5f}".format(l))
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Iterations")

    ax1.scatter(x, y, label = "data")
    ax1.plot(x, pred, c = 'r', label = "Regression Line")
    ax1.legend(loc="upper left")
    ax2.plot(list(range(j)), loss)
    fig.canvas.draw()


"""

x = [1, 2, 4, 3, 5]
y = [1, 3, 3, 2, 5]


def MSELoss(predicted, target):
    dif = (predicted - target) ** 2
    return np.mean(dif)

def GradientDecent(theta, x, predicted, target, learning_rate = 0.01):
    dif = predicted - target
    difprod = np.multiply(dif, x)
    gradient = np.mean(difprod, axis = 0)
    change = gradient.reshape(-1, 1) * learning_rate
    return theta - change

x_val = np.ones(shape=(len(x), 2))
y_val = np.array(y).reshape(-1, 1)
x_val[:, 1] = np.array(x).T

weights = np.random.rand(2, 1)

fig = plt.gcf()
fig.show()
fig.canvas.draw()

for j in range(100):
    pred = np.matmul(x_val, weights)
    print("Iteration {}, Loss : {:.6f}".format(j + 1, MSELoss(pred, y_val)))
    weights = GradientDecent(weights, x_val, pred, y_val)

    fig.clear()
    plt.scatter(x, y)
    plt.plot(x, pred, c = 'r')
    fig.canvas.draw()

"""
