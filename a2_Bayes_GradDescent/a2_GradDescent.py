import numpy as np

# Initialize Data Params
n = 1506
d = 5
learning_rate = 0.01

# Create data
x_data = np.random.rand(n, d)
x_data_1 = np.append(np.ones((n, 1)), x_data, axis=1)                                       # n x d+1
x_data_T = np.transpose(x_data_1)                                                           # d+1 x n
w_actual = np.random.rand(d+1, 1)                                                           # d+1 x 1
y_data = np.dot(x_data_1, w_actual)                                                         # (n x d+1) * (d+1 x 1) = n x 1
w_star = np.dot(np.dot(np.linalg.inv((np.dot(x_data_T, x_data_1))), x_data_T), y_data)      # d+1 x 1

# Weight grad
def calc_delt_w(x, y, w):
    p = np.dot(x, w)
    delt_w = 2*np.dot(np.transpose(x), y-p)
    return delt_w

# Weight second grad
def calc_delt_w2(x, y, w):
    sq_den = np.sum(np.square(w))
    sqrt_sq_den = np.sqrt(sq_den)
    n_1 = np.dot(x, w)
    term_1 = n_1/sqrt_sq_den
    term_2 = 2*n_1/sq_den
    delt_w = 2*np.dot(np.transpose(x), term_1 - term_2)
    return delt_w

# R2 accuracy calculation
def calc_R2(w_gd, w_star):
    num = np.sum(np.square(w_star - w_gd))
    y_m = np.mean(w_gd)
    den = np.sum(np.square(w_gd - y_m))
    return 1 - num/den

# Now we want to estimate our weight vector
w_init = np.random.rand(d+1, 1)           # initialize to 0 vector
num_epochs = 100
for ep in range(num_epochs):
    if ep == 0:
        w = w_init
    delt_w = calc_delt_w2(x_data_1, y_data, w)
    w -= learning_rate*delt_w
r2 = calc_R2(w, w_star)

print("W actual:")
print(w_actual)
print("\nW grad descent:")
print(w)
print("\nW*:")
print(w_star)
print(f"\nR squared: {r2}")



def calc_p(w, x):
    # calc p matrix
    n, d = x.shape
    p = np.empty((n,))
    for i in range(n):
        # num data points
        sum_ij = 0
        for j in range(d):
            sum_ij += w[j]*x[i, j]
            # num features
        # print(sum_ij)
        p[i,] = sum_ij
    p_diag = np.diag(p)
    return p, p_diag

def calc_e(x, y, p):
    n = x.shape[0]
    e = np.empty((n,))
    for i in range(n):
        e[i,] = p[i,] - y[i]
    return e

def calc_w(x, p_diag, e, learning_rate):
    n, d = x.shape
    delt_w = learning_rate*(np.dot(np.dot(np.dot(np.transpose(x), p_diag), np.identity(n) - p_diag), e))
    return delt_w.reshape(d,1)