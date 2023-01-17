import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import metrics as Metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Fix randomness
random.seed(1234)
np.random.seed(1234)


class Prob1():
    def __init__(self, info):
        self.info = info
        self.n_finegrid = info['n_finegrid']
        self.part = info['part']

        # Square bounds for each part
        if self.part == "A":
            self.bounds = np.array([[-4, -1, 3, 0], 
                                    [-2, 1, -1, -4], 
                                    [2, 5, 1, -2]])
        elif self.part == "B":
            self.bounds = np.array([[-4, -3, 3, 2], 
                                    [-1, 0, -2, -3], 
                                    [2, 3, 0, -1]])
        self.num_epochs = info['num_epochs']
        self.batch_size = info['batch_size']
        self.final_results = {}

    def generate_data(self):
        """
        Generate data --> random points with noisy class labels
        """
        # Both dims
        x1 = np.random.uniform(-6, 6 + 1e-10, self.n).reshape(self.n, 1)
        x2 = np.random.uniform(-4, 4 + 1e-10, self.n).reshape(self.n, 1)
        self.x_data = np.concatenate((x1, x2), axis=1)
        y = []
        for i in range(self.n):
            inSquare = False
            for pt in range(self.bounds.shape[0]):
                if self.bounds[pt, 0] < x1[i, 0] < self.bounds[pt, 1] and self.bounds[pt, 3] < x2[i, 0] < self.bounds[pt, 2]:
                    # Within a square
                    inSquare = True
                    if random.random() > 0.97:
                        # Incorrectly labeles as -
                        y.append(0)
                    else:
                        # Correctly labeled as +
                        y.append(1)
            if inSquare is False:
                # Not within a square
                if random.random() > 0.01:
                    # Correctly labeled as -
                    y.append(0)
                else:
                    # Incorrectly labeled as +
                    y.append(1)
        self.y_data = np.array(y).reshape(self.n, 1)
        self.all_data = np.concatenate((self.x_data, self.y_data), axis=1)
        return self.x_data, self.y_data

    def generate_finegrid_data(self):
        """
        Finegrid data. No noise class labels
        """
        x1 = np.random.uniform(-6, 6 + 1e-10, self.n_finegrid).reshape(self.n_finegrid, 1)
        x2 = np.random.uniform(-4, 4 + 1e-10, self.n_finegrid).reshape(self.n_finegrid, 1)
        self.x_data_finegrid = np.concatenate((x1, x2), axis=1)
        y = []
        for i in range(self.n_finegrid):
            inSquare = False
            for pt in range(self.bounds.shape[0]):
                if self.bounds[pt, 0] < x1[i, 0] < self.bounds[pt, 1] and self.bounds[pt, 3] < x2[i, 0] < self.bounds[pt, 2]:
                    # Within a square
                    inSquare = True
                    y.append(1)
            if inSquare is False:
                y.append(0)
        self.y_data_finegrid = np.array(y).reshape(self.n_finegrid, 1)
        self.all_data_finegrid = np.concatenate((self.x_data_finegrid, self.y_data_finegrid), axis=1)
        return self.x_data_finegrid, self.y_data_finegrid
    
    def split_data(self):
        """
        Split data in to test and train
        """
        indexes = np.arange(self.all_data.shape[0])
        np.random.shuffle(indexes)
        self.num_train = int(0.7*self.x_data.shape[0])
        self.train_data = self.all_data[indexes[:self.num_train], :]
        self.test_data = self.all_data[indexes[self.num_train:], :]
        self.x_train = self.train_data[:, [0, 1]]
        self.y_train = self.train_data[:, 2]
        self.x_test = self.test_data[:, [0, 1]]
        self.y_test = self.test_data[:, 2]

    def regularize(self):
        """
        Regularize data
        """
        self.x_data[:, 0] /= 6
        self.x_data[:, 1] /= 6
    
    def calculate_balanced_acc(self, FN, FP, TN, TP):
        """
        Balanced ACC calculation
        """
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        return (sensitivity + specificity)/2

    def NN(self, h1, h2):
        """
        Neural network method. Creates neural network with certain sized layers (3 or 4).
        """
        if h2 == 0:
            model = Sequential()
            model.add(Dense(h1, input_shape=(2,), activation=tf.keras.activations.tanh))
            model.add(Dense(1, activation=tf.keras.activations.tanh))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 
                                                                                tf.keras.metrics.AUC(), 
                                                                                tf.keras.metrics.FalseNegatives(), 
                                                                                tf.keras.metrics.FalsePositives(),
                                                                                tf.keras.metrics.TrueNegatives(),
                                                                                tf.keras.metrics.TruePositives()])
            model.fit(self.x_train, self.y_train, epochs=self.num_epochs, batch_size=self.batch_size)
            _, class_accuracy, auc, FN, FP, TN, TP = model.evaluate(self.x_test, self.y_test) #batch_size=self.x_test.shape[0]
            balanced_acc = self.calculate_balanced_acc(FN, FP, TN, TP)
            _, true_performance, _, _, _, _, _ = model.evaluate(self.x_data_finegrid, self.y_data_finegrid, batch_size=self.x_data_finegrid.shape[0])
        else:
            model = Sequential()
            model.add(Dense(h1, input_shape=(2,), activation=tf.keras.activations.tanh))
            model.add(Dense(h2, activation=tf.keras.activations.tanh))
            model.add(Dense(1, activation=tf.keras.activations.tanh))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 
                                                                                tf.keras.metrics.AUC(), 
                                                                                tf.keras.metrics.FalseNegatives(), 
                                                                                tf.keras.metrics.FalsePositives(),
                                                                                tf.keras.metrics.TrueNegatives(),
                                                                                tf.keras.metrics.TruePositives()])
            model.fit(self.x_train, self.y_train, epochs=self.num_epochs, batch_size=self.batch_size)
            _, class_accuracy, auc, FN, FP, TN, TP = model.evaluate(self.x_test, self.y_test)    #batch_size=self.x_test.shape[0]
            balanced_acc = self.calculate_balanced_acc(FN, FP, TN, TP)
            _, true_performance, _, _, _, _, _ = model.evaluate(self.x_data_finegrid, self.y_data_finegrid, batch_size=self.x_data_finegrid.shape[0])
        return (class_accuracy, balanced_acc, auc, true_performance)

    def save_data(self, results, h1, h2):
        """
        Save results for certain network dimensions
        """
        class_accuracy, balanced_acc, auc, true_performance = results
        name = str(self.n) + "_" + str(h1) + "_" + str(h2)
        self.final_results.update({name: [class_accuracy, balanced_acc, auc, true_performance]})

    def display_data(self):
        """
        Print data for final report
        """
        print("\n\n=========================")
        print("===== FINAL RESULTS =====")
        print("=========================")
        for k, v in self.final_results.items():
            n = k.split('_')[0]
            h1 = k.split('_')[1]
            h2 = k.split('_')[2]
            print(f"N: {n}, H1: {h1}, H2: {h2}")
            print(f"\tClassification Accuracy: {round(v[0]*100, 2)}%")
            print(f"\tBalanced Accuracy: {round(v[1]*100, 2)}%")
            print(f"\tROC AUC Estimate: {round(v[2], 3)}")
            print(f"\tTrue Performance: {round(v[3]*100, 2)}%\n")

    def run(self):
        """
        Run everything
        """
        x_fg, y_fg, = self.generate_finegrid_data()
        for n in [250, 1_000, 10_000]:
            self.n = n
            x, y = self.generate_data()
            self.split_data()
            for h1 in [1, 4, 12]:
                for h2 in [0, 3]:
                    results = self.NN(h1, h2)
                    self.save_data(results, h1, h2)
                    print("\n\n==========================================")
                    print(f"===== DONE WITH N: {n} H1: {h1} H2: {h2}")
                    print("==========================================\n\n")
        self.display_data()



# Run Part A
params = {"n_finegrid": 1_000_000,
        "part": "A",
        "num_epochs": 150,
        "batch_size": 32}
prob = Prob1(params)
prob.run()

# Run Part B
params = {"n_finegrid": 1_000_000,
        "part": "B",
        "num_epochs": 150,
        "batch_size": 32}
prob = Prob1(params)
prob.run()