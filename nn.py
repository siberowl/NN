import numpy as np
import pandas as pd


class NeuralNet:
    def __init__(self, num_feature, num_hidden, num_classification):
        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.num_classification = num_classification
        self.input_weights = np.random.rand(self.num_hidden,
                                            self.num_feature)
        self.output_weights = np.random.rand(self.num_classification,
                                             self.num_hidden)
        self.learning_rate = 0.1

    def train(self, input, target):
        hidden = self.predict_hidden(input)
        output = self.predict_output(hidden)
        r = self.learning_rate
        output_errors = self.output_errors(output, target)
        input_errors = self.input_errors(output_errors,
                                         self.output_weights,
                                         hidden)

        self.output_weights += r * output_errors.reshape(-1, 1) * np.tile(
                hidden, (len(output_errors), 1))
        self.input_weights += r * input_errors.reshape(-1, 1) * np.tile(
                input, (len(input_errors), 1))

    def output_errors(self, output, target):
        return (target - output) * output * (np.ones(len(output)) - output)

    def input_errors(self, output_errors, output_weights, hidden):
        sums = np.dot(output_errors, output_weights)
        return sums * np.array(hidden * (np.ones(len(hidden)) - hidden))

    def predict_hidden(self, input):
        return self.predict_layer(input, self.input_weights)

    def predict_output(self, hidden):
        return self.predict_layer(hidden, self.output_weights)

    def predict_layer(self, input, weights):
        return self.activation(np.dot(weights, input))

    def predict(self, sample):
        result = self.predict_output(self.predict_hidden(sample))
        index = np.argmax(result)
        return index

    def weighted_sum(self, inputs, weights):
        return np.sum(inputs * weights)

    def activation(self, xs):
        sigmoid = np.vectorize(lambda x: 1/(1 + np.exp(-1 * x)))
        return sigmoid(xs)


n_training_set = 130

data = pd.read_csv('Iris.csv', encoding='utf-8')
data = data.replace(to_replace="Iris-setosa", value=0)
data = data.replace(to_replace="Iris-versicolor", value=1)
data = data.replace(to_replace="Iris-virginica", value=2)
data = data.sample(frac=1)

training_set = data.iloc[:n_training_set, 1:5].values
training_labels = data.iloc[:n_training_set, 5].values
training_classifications = np.zeros(n_training_set * 3, dtype=int).reshape(n_training_set, 3)
training_classifications[np.arange(n_training_set), training_labels] = 1

prediction_set = data.iloc[n_training_set:150, 1:5].values
prediction_labels = data.iloc[n_training_set:150, 5].values
prediction_classifications = np.zeros(20 * 3, dtype=int).reshape(20, 3)
prediction_classifications[np.arange(20), prediction_labels] = 1


model = NeuralNet(4, 5, 3)

nepoch = 100

for n in range(nepoch):
    for t in range(n_training_set):
        model.train(training_set[t], training_classifications[t])

count = 0
errors = 0
for p in range(len(prediction_set)):
    prediction = model.predict(prediction_set[p])
    actual = np.argmax(prediction_classifications[p])
    print('p: ' + str(prediction) + ', a: ' + str(actual))
    if prediction != actual:
        errors+=1
    count += 1

print(1 - errors/count)
