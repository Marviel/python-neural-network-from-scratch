from random import seed
from random import random
from math import exp


class Neuron(object):
    def __init__(self, bias, weights, output=[]):
        self.bias = bias
        self.weights = weights
        self.output = output

    def __repr__(self):
        return f"Neuron({str(self.__dict__)})"


class DataSample(object):
    def __init__(self, data_in, data_out):
        self.data_in = data_in
        self.data_out = data_out


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    hidden_layer = [Neuron(bias=random(), weights=[random()
                                                   for i in range(n_inputs)]) for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [Neuron(bias=random(), weights=[random()
                                                   for i in range(n_hidden)]) for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(bias, weights, inputs):
    activation = float(bias)
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]

    return activation

# Sigmoid activation function, the traditional activation function


def transfer_sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


def transfer_sigmoid_derivative(output):
    return output * (1.0 - output)


def forward_propagate(network, row):
    inputs = row

    for layer in network:
        new_inputs = []

        for neuron in layer:
            activation = activate(neuron.bias, neuron.weights, inputs)
            neuron.output = transfer_sigmoid(activation)
            new_inputs.append(neuron.output)

        inputs = new_inputs

    return inputs


def backward_propagate_error(network, expected):
    # Go through layers backwards.
    for layer_i in reversed(range(len(network))):
        layer = network[layer_i]

        errors = []

        # If NOT the last layer
        if layer_i != len(network) - 1:
            for neuron_i in range(len(layer)):
                error = 0.0

                # Go through the neurons in the layer AFTER this one,
                # and calculate the error of this neuron.
                for neuron in network[layer_i + 1]:
                    error += (neuron.weights[neuron_i] * neuron.delta)

                # Add to the errors for this layer.
                errors.append(error)

        # If IS the last layer
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron.output)

        # Go through all neurons in this layer,
        # And determine their delta using error * derivative(activation_function)
        for j, neuron in enumerate(layer):
            neuron.delta = errors[j] * \
                transfer_sigmoid_derivative(neuron.output)


# Using each neuron's delta calculated via backpropagation,
# we can now update the weights according to the input.
def update_weights(network, row, l_rate):
    inputs = row
    for i in range(len(network)):
        # For all layers but the first,
        # we make the inputs the previous layer's outputs.
        if i != 0:
            inputs = [neuron.output for neuron in network[i - 1]]

        # For every neuron in this layer,
        # We use the delta (calculated during backpropagation), the input that caused the error, and the learning rate,
        # to update its weights, and biases.
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron.weights[j] += l_rate * neuron.delta * inputs[j]

            neuron.bias += l_rate * neuron.delta


def train_network(network, train_data, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        # For each training data sample
        for sample in train_data:
            # Calculate new outputs
            outputs = forward_propagate(network, sample.data_in)

            # This is a classification problem, so we expect the output
            # to match the class this data sample corresponds to.
            # By our convention, this class is given by the final element of
            # Each sample. Thus: expected[row[-1]] = 1 says that the expected
            # vector, which is otherwise zero, will be "hot" for the entry
            # this data sample is expecting.
            expected = [0 for i in range(n_outputs)]
            expected[sample.data_out] = 1

            # Calculate the summed error.
            sum_error += sum([(expected[i]-outputs[i]) **
                              2 for i in range(len(expected))])

            # Calculate the error for each neuron via backpropagation
            backward_propagate_error(network, expected)

            # Update the weights
            update_weights(network, sample.data_in, l_rate)

        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


if __name__ == "__main__":
    # seed(1)
    # network = initialize_network(2, 1, 2)

    # row = [1, 0]
    # output = forward_propagate(network, row)

    # print(output)

    # test backpropagation of error
    # network = [
    #     [
    #         Neuron(
    #             output=0.7105668883115941,
    #             weights=[0.13436424411240122, 0.8474337369372327],
    #             bias=0.763774618976614)
    #     ],
    #     [
    #         Neuron(
    #             output=0.6213859615555266,
    #             weights=[0.2550690257394217],
    #             bias=0.49543508709194095
    #         ),
    #         Neuron(
    #             output=0.6573693455986976,
    #             weights=[0.4494910647887381],
    #             bias=0.651592972722763
    #         )
    #     ]
    # ]
    # expected = [0, 1]
    # backward_propagate_error(network, expected)
    # for layer in network:
    #     print(layer)

    seed(1)
    # [[Input, Input, Expected Output]...]
    dataset = [
        DataSample([2.7810836, 2.550537003], 0),
        DataSample([1.465489372, 2.362125076], 0),
        DataSample([3.396561688, 4.400293529], 0),
        DataSample([1.38807019, 1.850220317], 0),
        DataSample([3.06407232, 3.005305973], 0),
        DataSample([7.627531214, 2.759262235], 1),
        DataSample([5.332441248, 2.088626775], 1),
        DataSample([6.922596716, 1.77106367], 1),
        DataSample([8.675418651, -0.242068655], 1),
        DataSample([7.673756466, 3.508563011], 1)
    ]
    n_inputs = len(dataset[0].data_in)
    n_outputs = len(set([row.data_out for row in dataset]))

    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, 0.5, 40, n_outputs)
    for layer in network:
        print(layer)
