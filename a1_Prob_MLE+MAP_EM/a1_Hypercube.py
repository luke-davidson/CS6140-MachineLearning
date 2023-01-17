import numpy as np
import math
import matplotlib.pyplot as plt

n = 8
d = range(1, 101)
percents = []
percents_master = []

def GenerateDataPoints(n, d):
    """
    Generate hypersphere data points of dimension d
    """
    data = np.random.rand(n, d)
    return data

def PercentInHypersphere(d):
    """
    Calc percent of hypercube in hypersphere
    """
    percent = math.pi**(d/2)/(math.gamma(((d/2)+1))*2**d)
    return percent

# Determine number of generated data points within hypersphere
for dim in d:
    percent = PercentInHypersphere(dim)
    data = GenerateDataPoints(n, dim)
    percents = []
    for points in range(n):
        count = 0
        for pt in data[points, :]:
            if pt < percent:
                count += 1
        percents.append(count/dim)
    print(percents)
    percents_master.append(sum(percents)/n)

plt.plot(d, percents_master)
plt.grid()
plt.xlabel("Dimension")
plt.ylabel("Percent of data points within the hypersphere")
plt.title(f"Luke Davidson Q8c - n={n}")
plt.show()