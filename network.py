from . import neuron as n
import random
import numpy as np
import copy 

class Network:

    def __init__(self):
        self.neurons = np.array([])
        self.activations = np.array([])
        self.connections = np.array([], ndmin=2, dtype=float)
        self.num_neurons = 0

    def _neurons_to_indices(self, neuron_array):
        if not isinstance(neuron_array, list):
            neuron_array = [neuron_array]
        return [n.index for n in neuron_array]


    def _create_neuron_in_array(self, ntype, **kwargs):
        new_index = len(self.neurons)
        neuron = ntype(network=self, index=new_index, **kwargs)
        self.neurons = np.append(self.neurons, neuron)
        return neuron

    def _add_neurons_to_connection_matrix(self, num_neurons):
        self.num_neurons += num_neurons
        self.connections = np.resize(self.connections, (self.num_neurons, self.num_neurons))

    def _init_activation(self):
        self.activations = [i.get_activation() for i in self.neurons]


    def create_neuron(self, ntype=None, **kwargs):
        neuron = self._create_neuron_in_array(ntype, **kwargs)
        self._add_neurons_to_connection_matrix(1)
        return neuron


    def create_neuron_array(self, size=3, ntype=None, **kwargs):
        neurons = []
        name = None
        if "name" in kwargs:
            name = kwargs["name"]
        for i in range(size):
            if name:
                kwargs["name"] = name + "_" + str(i)

            neuron = self._create_neuron_in_array(ntype, **kwargs)
            neurons.append(neuron)

        self._add_neurons_to_connection_matrix(size)
        return neurons


    def set_activation(self, index, value):
        try:
            self.activations[index] = value
        except IndexError:
            self._init_activation()
            self.activations[index] = value

    def _set_neuron_connection(self, source, targets, value):
        if not (isinstance(value, int) or isinstance(value, float)):
            value = random.random()
        self.connections[source.index, [t.index for t in targets]] = value

    def set_connections(self, sources, targets, value):
        if not isinstance(sources, list):
            sources = [sources] 
        if not isinstance(targets, list):
            targets = [targets]
        for n in sources:
            self._set_neuron_connection(n, targets, value)

    def update_connection_strength(self, input_n, output_n, new_value):
        input_n = self._neurons_to_indices(input_n)
        output_n = self._neurons_to_indices(output_n)

        self.connections[input_n, output_n] = new_value
        self.connections[output_n, input_n] = new_value

    def increase_connection_strength(self, input_n, output_n, value):
        input_n = self._neurons_to_indices(input_n)
        output_n = self._neurons_to_indices(output_n)

        self.connections[input_n, output_n] += value
        self.connections[output_n, input_n] += value


    def all_to_all_connectivity(self, neurons=None, connection_strength=0):
        if not neurons:
            neurons = self.neurons

        for i in range(len(neurons) - 1):
            neurons[i].listen_to(neurons[i+1], connection_strength)
            neurons[i].send_to(neurons[i+1], connection_strength)


    def set_lifespan(self, neuron, lifespan):
        if isinstance(neuron, list):
            for single_neuron in neuron:
                self.set_lifespan(single_neuron, lifespan)
            return

        neuron.set_lifespan(lifespan)


    def apply_pattern(self, pattern):
        for neuron in self.neurons:
            if neuron in pattern:
                neuron.set_activation(1)
            else:
                neuron.set_activation(-1)


    def apply_noise(self, size=0.01):
        for neuron in self.neurons:
            noise = (random.random() - 0.5) * size
            neuron.set_activation(neuron.get_activation() + noise)


    def run(self):
        for neuron in self.neurons:
            neuron.run()


    @property
    def neuron_list(self):
        return self.neurons


    # TODO: DRY in the next two methods
    def run_and_get_activations(self, neurons=None, activations=None, steps=80):

        if not neurons:
            neurons = self.neuron_list

        self._init_activation()

        activations = []
        for i in range(steps):

            inputs = np.mat(self.connections) * np.mat(np.transpose(self.activations)).T

            for neuron in neurons:
                neuron.add_input(inputs[neuron.index])

            activations.append(copy.deepcopy(self.activations))

            self.run()
                
        return np.mat(activations)


    def run_and_get_results(self, func, func_steps=10, results=None, steps=80):

        if not results:
            results = []

        self._init_activation()

        activations = []
        for i in range(steps):

            inputs = np.mat(self.connections) * np.mat(np.transpose(self.activations)).T

            if func and  i % func_steps == 0:
                results.append(func(self.neurons))

            activations.append(copy.deepcopy(self.activations))

            self.run()

        return results, np.mat(activations)
        

    def run_and_get_phases(self, neurons=None, steps=80):

        if not neurons:
            neurons = self.neuron_list
        
        activations = {}
        for neuron in neurons:
            activations[neuron] = []
        
        for i in range(steps):
            for neuron in neurons:
                activations[neuron].append(neuron.get_activation())

            self.run()
                
        return activations

