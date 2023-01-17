import numpy as np

part = "a"

# dist to hyperplane
def dist(w, x):
    num = np.dot(w, x)[0]
    den = np.sqrt(np.sum(np.square(w)))
    return num/den

if part == "b":
    # weights and data points
    w = np.array([0.3, 0.5, 0.6]).reshape(1, 3)
    x = np.array([1, 0.1, 0.8]).reshape(3, 1)

    dist_1 = dist(w, x)
    print(f"First distance: {dist_1[0]}")

    # incorrect data point weight is updated
    w[0, 1] -= x[1, 0]

    dist_2 = dist(w, x)
    print(f"Updated weight distance: {dist_2[0]}")
else:
    # Part a
    num_tests = 1000
    results_add = []
    results_sub = []
    for pn in range(2):
        """
        0: add
        1: sub
        """
        num_dec_underclassified = 0
        num_inc_overclassified = 0
        # Calculate number of incorrectly classified points in the hyperplane
        for i in range(num_tests):
            w = np.random.rand(1,3)
            val = -100
            x = np.array([1, np.random.random()*val, np.random.random()*val]).reshape(3, 1)
            dist_1 = dist(w, x)
            if pn == 0:
                w[0, np.random.randint(1,3)] += x[np.random.randint(1,3), 0]
                dist_2 = dist(w, x)
                if dist_1 > dist_2:
                    # decreased when underclassified
                    num_dec_underclassified += 1
            else:
                w[0, np.random.randint(1,3)] -= x[np.random.randint(1,3), 0]
                dist_2 = dist(w, x)
                if dist_1 < dist_2:
                    # increased when overclassified
                    num_inc_overclassified += 1
        if pn == 0:
            print(num_dec_underclassified)
        else:
            print(num_inc_overclassified)
