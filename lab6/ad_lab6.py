import numpy as np
import matplotlib.pyplot as plt

k = 2
b = 1

# n_points = 100

# Генерація випадкових значень x
x = np.linspace(-10, 10, 100)

# Додавання нормального шуму до y
noise = np.random.normal(0, 2, size=x.shape)  # середнє 0, стандартне відхилення 2

# Обчислення y за формулою прямої + шум
y = k * x + b + noise
x_sorted = np.sort(x)
true_y = k * x_sorted + b

def least_squares(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    # Обчислення середніх значень
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Обчислення коефіцієнта k
    chuselnyk = np.sum((x - x_mean) * (y - y_mean))
    znamennyk = np.sum((x - x_mean) ** 2)
    my_k = chuselnyk / znamennyk

    # Обчислення коефіцієнта b
    my_b = y_mean - my_k * x_mean

    return my_k, my_b

my_k, my_b = least_squares(x, y)
my_y = my_k * x_sorted + my_b

def numpy_least_squares(x, y):
    a = np.polyfit(x,y,1)
    return a[0], a[1]

numpy_k, numpy_b = numpy_least_squares(x, y)
numpy_y = numpy_k * x_sorted + numpy_b




def gradient_descent(x, y, learning_rate=0.01, n_iter=1000):
    k = 0.0
    b = 0.0
    n = len(x)
    mse_history = []

    for i in range(n_iter):
        y_pred = k * x + b
        error = y_pred - y
        mse = np.mean(error ** 2)
        mse_history.append(mse)

        # Градієнти
        dk = (2/n) * np.dot(error, x)
        db = (2/n) * np.sum(error)

        # Оновлення параметрів
        k -= learning_rate * dk
        b -= learning_rate * db

    return k, b, mse_history



n_iter = 200
grad_k, grad_b, mse_history = gradient_descent(x, y, learning_rate=0.01, n_iter=n_iter)
grad_y = grad_k * x_sorted + grad_b



fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(x, y, color='blue', label='Точки (x, y)', alpha=0.7)
axs[0].plot(x_sorted, true_y, color='red', label=f'Справжня пряма: y = {k}x + {b}')
axs[0].plot(x_sorted, my_y, color='blue', label=f'МНК вручну: y = {my_k:.2f}x + {my_b:.2f}')
axs[0].plot(x_sorted, numpy_y, color='green', linestyle='--', label=f'polyfit: y = {numpy_k:.2f}x + {numpy_b:.2f}')
axs[0].plot(x_sorted, grad_y, color='black', linestyle='--', label=f'Градієнт: y = {grad_k:.2f}x + {grad_b:.2f}')
axs[0].set_title('Регресії')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].grid(True)

# Графік 2: збіжність похибки
axs[1].plot(range(n_iter), mse_history, color='orange')
axs[1].set_title('Збіжність градієнтного спуску')
axs[1].set_xlabel('Ітерації')
axs[1].set_ylabel('MSE')
axs[1].grid(True)

plt.tight_layout()
plt.show()
