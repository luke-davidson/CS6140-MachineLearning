import math
import random
import numpy as np
import matplotlib.pyplot as plt

class EM():
    """
    Implementation of the EM algorithm
    """
    def __init__(self, mn, mu1_actual, mu2_actual, sig1_actual, sig2_actual, alpha_actual, beta_actual):
        """
        Pass in actual parameter values that the parameter estimates should converge to
        """
        self.mn = mn
        self.mu1_actual = mu1_actual
        self.mu2_actual = mu2_actual
        self.sig1_actual = sig1_actual
        self.sig2_actual = sig2_actual
        self.alpha_actual = alpha_actual
        self.beta_actual = beta_actual
        self.LL = np.empty((0,1))

    def initialize(self, mu1_init, mu2_init, sig1_init, sig2_init, alpha_init, beta_init):
        """
        Initialize parameters to initial parameter estimates
        """
        self.mu1_est = mu1_init
        self.mu2_est = mu2_init
        self.sig1_est = sig1_init
        self.sig2_est = sig2_init
        self.alpha_est = alpha_init
        self.beta_est = beta_init

    def PD_d(self, d, ab, k) -> float:
        """
        Args:
            d: float, x value?
            ab: string; "alpha" or "beta"
            k: int; 1 or 2 (which mixture)
        Returns:
            float; value of pD(d) for the given data point
        """
        if ab == "alpha":
            if k == 1:
                return self.alpha_est*(math.exp((-(d-self.mu1_est)**2)/(2*(self.sig1_est)**2))/(math.sqrt(2*(math.pi)*(self.sig1_est)**2)))
            if k == 2:
                return (1-self.alpha_est)*(math.exp((-(d-self.mu2_est)**2)/(2*(self.sig2_est)**2))/(math.sqrt(2*(math.pi)*(self.sig2_est)**2)))
        elif ab == "beta":
            if k == 1:
                return self.beta_est*(math.exp((-(d-self.mu1_est)**2)/(2*self.sig1_est**2))/(math.sqrt(2*(math.pi)*(self.sig1_est**2))))
            if k == 2:
                return (1-self.beta_est)*(math.exp((-(d-self.mu2_est)**2)/(2*self.sig2_est**2))/(math.sqrt(2*(math.pi)*(self.sig2_est**2))))

    def generateNormal(self, num, mode):
        """
        Args:
            num: int; 1 or 2
            mode: string; "est" or "actual"
        Returns:
            np.array; normal distribution based on given inputs
        """
        if num == 1:
            if mode == "est":
                return np.random.normal(self.mu1_est, self.sig1_est, self.mn)
            elif mode == "actual":
                return np.random.normal(self.mu1_actual, self.sig1_actual, int(self.mn/2))
        elif num == 2:
            if mode == "est":
                return np.random.normal(self.mu2_est, self.sig2_est, self.mn)
            elif mode == "actual":
                return np.random.normal(self.mu2_actual, self.sig2_actual, int(self.mn/2))

    def generateData(self, xy):
        """
        Generates initial data based on actual values
        Args:
            xy: string; "x" or "y"
        Returns:
            np.array: x or y data based on input
        """
        if xy == 'x':
            x_dist_1 = self.alpha_actual*self.generateNormal(1, "actual")
            x_dist_2 = (1-self.alpha_actual)*self.generateNormal(2, "actual")
            self.x_data = np.concatenate((x_dist_1, x_dist_2)).flatten()
        elif xy == 'y':
            y_dist_1 = self.beta_actual*self.generateNormal(1, "actual")
            y_dist_2 = (1-self.beta_actual)*self.generateNormal(2, "actual")
            self.y_data = np.concatenate((y_dist_1, y_dist_2)).flatten()
    
    def E_step(self):
        # Initialize arrays to hold pseudo posteriors
        self.og_gammas = np.empty((len(self.x_data),2))
        self.gammas = np.empty((len(self.x_data),2))
        for i in range(len(self.x_data)):
            for mode in [1, 2]:
                self.og_gammas[i, mode-1] = em.PD_d(self.x_data[i], "alpha", mode)
        self.gammas[:,0] = self.og_gammas[:,0]/np.sum(self.og_gammas, axis=1)
        self.gammas[:,1] = self.og_gammas[:,1]/np.sum(self.og_gammas, axis=1)
    
    def M_step(self):
        N_1, N_2 = np.sum(self.gammas, axis=0)
        self.mu1_est = (1/N_1)*np.sum(np.multiply(self.gammas[:,0], self.x_data))
        self.mu2_est = (1/N_2)*np.sum(np.multiply(self.gammas[:,1], self.x_data))
        self.sig1_est = math.sqrt((1/N_1)*np.sum((self.gammas[:,0])*np.square((self.x_data - self.mu1_est))))
        self.sig2_est = math.sqrt((1/N_2)*np.sum((self.gammas[:,1])*np.square((self.x_data - self.mu2_est))))
        self.alpha_est = N_1/self.mn
        self.beta_est = N_2/self.mn
    
    def computeLogLikelihood(self):
        self.LL = np.append(self.LL, np.sum(np.log(np.sum(self.og_gammas, axis=1))))



num_tests = 1000
max_steps = 100
results = np.empty((0,6))
results_main = np.empty((0,6))

for i in range(num_tests):
    em = EM(1000, 10, 12, 2, 0.5, 0.5, 0.7)
    em.initialize(1, 1, 1, 1, 0.1, 0.6)
    em.generateData('x')
    em.generateData('y')

    for step in range(max_steps):
        em.E_step()
        em.M_step()
        results = np.append(results, np.array([[em.mu1_est, em.mu2_est, em.sig1_est, em.sig2_est, em.alpha_est, em.beta_est]]), axis=0)
        em.computeLogLikelihood()
    results_main = np.append(results_main, np.array([[em.mu1_est, em.mu2_est, em.sig1_est, em.sig2_est, em.alpha_est, em.beta_est]]), axis=0)
    print(F"[INFO]: {i+1} / {num_tests} complete.")

# Post process data
averages = np.mean(results_main, axis=0)
stddevs = np.std(results_main, axis=0)

print(averages)
print(stddevs)

# Plots
x_range = range(max_steps)
est_labels = ["mu1 est", "mu2 est", "sigma1 est", "sigma2 est", "alpha est"]
actual = [em.mu1_actual, em.mu2_actual, em.sig1_actual, em.sig2_actual, em.alpha_actual]
actual_labels = ["mu1 actual", "mu2 actual", "sigma1 actual", "sigma2 actual", "alpha actual"]
colors = ["r", "b", "g", "c", "m"]
# for i in range(5):
#     plt.plot(x_range, [actual[i]]*len(x_range), colors[i], label = actual_labels[i])
#     plt.plot(x_range, results[:,i], colors[i], label = est_labels[i])
plt.plot(x_range, em.LL, label = "Log Likelihood")
plt.legend()
plt.title("Luke Davidson - Q7c")
plt.ylabel("Param Val")
plt.xlabel("Iteration")
plt.show()

# print(em.mu1_est, em.mu2_est, em.sig1_est, em.sig2_est, em.alpha_est)