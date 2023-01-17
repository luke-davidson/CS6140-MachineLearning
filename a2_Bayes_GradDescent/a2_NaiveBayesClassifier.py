import numpy as np

# data
x_data = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
y_labels = np.array([1, 0, 0, 1, 0, 1, 1, 0]).reshape(8,1)

# part b data
new_data = np.empty((8, 12))
for n in range(x_data.shape[0]):
    new_data[n, 0] = x_data[n, 0]                                       # x1
    new_data[n, 1] = x_data[n, 1]                                       # x2
    new_data[n, 2] = x_data[n, 2]                                       # x3
    new_data[n, 3] = x_data[n, 0] * x_data[n, 1]                        # x1 * x2
    new_data[n, 4] = x_data[n, 0] * x_data[n, 2]                        # x1 * x3
    new_data[n, 5] = x_data[n, 1] * x_data[n, 2]                        # x2 * x3
    new_data[n, 6] = x_data[n, 0] * x_data[n, 1] * x_data[n, 2]         # x1 * x2 * x3
    new_data[n, 7] = x_data[n, 0] * x_data[n, 0] * x_data[n, 1]         # x1 * x1 * x2
    new_data[n, 8] = x_data[n, 0] * x_data[n, 1] * x_data[n, 1]         # x1 * x2 * x2
    new_data[n, 9] = x_data[n, 0] * x_data[n, 2] * x_data[n, 2]         # x1 * x3 * x3
    new_data[n, 10] = x_data[n, 1] * x_data[n, 1] * x_data[n, 2]        # x2 * x2 * x3
    new_data[n, 11] = x_data[n, 1] * x_data[n, 2] * x_data[n, 2]        # x2 * x3 * x3

class Evaluator():
    """
    Class representative of a Naive Bayes classifier
    """
    def __init__(self, x_data, labels, part):
        """
        x_data: np.array with x data
        labels: np.array with y (labels)
        part: string "a" or "b"
        """
        if part == "a":
            self.x_data = x_data
        elif part == "b":
            # Higher dimension data for part b
            self.x_data = np.empty((8,12))
            for n in range(x_data.shape[0]):
                self.x_data[n, 0] = x_data[n, 0]                                       # x1
                self.x_data[n, 1] = x_data[n, 1]                                       # x2
                self.x_data[n, 2] = x_data[n, 2]                                       # x3
                self.x_data[n, 3] = x_data[n, 0] * x_data[n, 1]                        # x1 * x2
                self.x_data[n, 4] = x_data[n, 0] * x_data[n, 2]                        # x1 * x3
                self.x_data[n, 5] = x_data[n, 1] * x_data[n, 2]                        # x2 * x3
                self.x_data[n, 6] = x_data[n, 0] * x_data[n, 1] * x_data[n, 2]         # x1 * x2 * x3
                self.x_data[n, 7] = x_data[n, 0] * x_data[n, 0] * x_data[n, 1]         # x1 * x1 * x2
                self.x_data[n, 8] = x_data[n, 0] * x_data[n, 1] * x_data[n, 1]         # x1 * x2 * x2
                self.x_data[n, 9] = x_data[n, 0] * x_data[n, 2] * x_data[n, 2]         # x1 * x3 * x3
                self.x_data[n, 10] = x_data[n, 1] * x_data[n, 1] * x_data[n, 2]        # x2 * x2 * x3
                self.x_data[n, 11] = x_data[n, 1] * x_data[n, 2] * x_data[n, 2]        # x2 * x3 * x3
        self.labels = labels
        self.all_data = np.append(self.x_data, self.labels, axis=1)
        self.num_pts = self.x_data.shape[0]
        self.pos_data = np.empty((0, self.all_data.shape[1]))
        self.neg_data = np.empty((0, self.all_data.shape[1]))
        # Labels
        for pt in range(self.num_pts):
            if self.all_data[pt, -1] == 1:
                self.pos_data = np.append([self.all_data[pt, :]], self.pos_data, axis=0)
            else:
                self.neg_data = np.append([self.all_data[pt, :]], self.neg_data, axis=0)
        self.num_pos = self.pos_data.shape[0]
        self.num_neg = self.neg_data.shape[0]
        self.p_pos = self.num_pos/self.num_pts
        self.p_neg = self.num_neg/self.num_pts

    def evaluate(self, data_pts):
        """
        Calculate Bayes theorem
        data_pts: np.array of data points, shape (n, d)
        """
        self.results = np.empty((data_pts.shape[0], 2))

        for k in range(2):
            for n in range(data_pts.shape[0]):
                if k == 0:
                    self.neg = self.p_neg
                else:
                    self.pos = self.p_pos
                for d in range(data_pts.shape[1]):
                    p_pt_k = self.calculate_P(data_pts[n, d], k, d)
                    if k == 0:
                        self.neg *= p_pt_k
                    else:
                        self.pos *= p_pt_k
                if k == 0:
                    self.results[n, 0] = self.neg
                else:
                    self.results[n, 1] = self.pos
        self.calculate_error()

    def calculate_P(self, data_pt_val, k, d):
        """
        Calculates class posterior
        """
        if k == 0:
            count_neg = 0
            for row in range(self.neg_data.shape[0]):
                if self.neg_data[row, d] == data_pt_val:
                    count_neg += 1
            return count_neg/self.num_neg
        elif k == 1:
            count_pos = 0
            for row in range(self.pos_data.shape[0]):
                if self.pos_data[row, d] == data_pt_val:
                    count_pos += 1
            return count_pos/self.num_pos

    def calculate_error(self):
        """
        Calc total error and print results.
        """
        self.predictions = np.argmax(self.results, axis=1)
        print("Results from each test data point:")
        print(self.results)
        print("Predicted classes from test data:")
        print(self.predictions)
        print("Actual classes of test data:")
        print(self.labels.reshape(8,))
        count = 0
        for i in range(self.predictions.shape[0]):
            if self.predictions[i] == self.labels[i]:
                count += 1
        print(f"Accuracy: {round(count/self.predictions.shape[0]*100, 4)}%")

# Execute
env = Evaluator(x_data, y_labels, "a")
env.evaluate(x_data)

print("Positvely classified data:")
print(env.pos_data)
print("Negatively classified data:")
print(env.neg_data)