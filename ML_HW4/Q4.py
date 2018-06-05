############################
#                          #
# authors:                 #
# Zixi Huang(zh2313)       # 
# Neil Kumar(nk2739)       # 
# Yichen Pan(yp2450)       #
#                          #
############################

import numpy as np
import matplotlib.pyplot as plt

number_of_cities = 9
cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']

# random coordinates
x = np.random.random([number_of_cities, 2])

# distance
d = np.array(
    [
        [0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
        [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
        [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
        [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
        [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
        [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
        [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
        [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
        [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]
    ]
)

# set parameter
iterations = 5000
alpha = 0.01

# initialize gradient
gradient = np.ones([number_of_cities, 2])

# control learning rate
count = 0
old_loss = 0

loss_list = []

for iteration in range(iterations):
    # compute gradient
    for city_index in range(number_of_cities):
        a = - 4 * np.sum(x - x[city_index], axis=0)
        b = 4 * (x - x[city_index]) * d[city_index].reshape([number_of_cities, -1])
        c = np.sqrt(np.sum((x - x[city_index]) ** 2, axis=1)).reshape([number_of_cities, -1])
        c[city_index] = 1

        gradient[city_index] = a + np.sum(b / c, axis=0)

    # update
    x -= alpha * gradient

    # record loss
    loss = 0

    for city_index in range(number_of_cities):
        loss += np.sum(
            (np.sqrt(np.sum((x - x[city_index]) ** 2, axis=1)).reshape([number_of_cities, -1])
             - d[city_index].reshape([number_of_cities, -1]))**2
        )

    if loss >= old_loss:
        count += 1
        if count > 10:
            count = 0
            alpha /= 2
    if loss < old_loss:
        alpha = 0.01

    old_loss = loss
    loss_list.append(loss)

print("gradient", gradient)
print("loss", loss)

# plot
plt.figure()
plt.scatter(x[:, 0], x[:, 1])
for index, coordinate in enumerate(x):
    plt.annotate(cities[index], (coordinate[0], coordinate[1]))

plt.figure()
plt.plot(loss_list)

plt.show()
