from . import neuron as n
import random

class Network:

    def __init__(self):
        self.neurons = []

    def update_connection_strength(input_n, output_n, new_value):
        if isinstance(input_n, list):
            for single_neuron in input_n:
                self.update_connection_strength(single_neuron, new_value)
            return

        if isinstance(output_n, list):
            for single_neuron in output_n:
                self.update_connection_strength(single_neuron, new_value)
            return

        input_n.update_connection_strength(output_n, new_value)

    def create_neuron(self, ntype = None, **kwargs):
        neuron = ntype(**kwargs)

        self.neurons.append(neuron)
        return neuron

    def create_neuron_array(self, size=3, ntype=None, **kwargs):
        neurons = []
        name = None
        if "name" in kwargs:
            name = kwargs["name"]
        for i in range(size):
            if name:
                kwargs["name"] = name + "_" + str(i)

            neurons.append(self.create_neuron(ntype, **kwargs))
        return neurons

    def set_lifespan(self, neuron, lifespan):
        if isinstance(neuron, list):
            for single_neuron in neuron:
                self.set_lifespan(single_neuron, lifespan)
            return

        neuron.set_lifespan(lifespan)

    def all_to_all_connectivity(self, neurons=None, connection_strength=0):
        if not neurons:
            neurons = self.neurons

        for i in range(len(neurons) - 1):
            neurons[i].listen_to(neurons[i+1], connection_strength)
            neurons[i].send_to(neurons[i+1], connection_strength)

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

    def run_and_get_activations(self, neurons=None, activations=None, steps=80):

        if not neurons:
            neurons = self.neuron_list
        
        if not activations:
            activations = {}
            for neuron in neurons:
                activations[neuron] = []

        else:
            for neuron in neurons:
                if neuron not in activations:
                    activations[neuron] = []


        for i in range(steps):
            for neuron in neurons:
                activations[neuron].append(neuron.get_activation())

            self.run()
                
        return activations

    def run_and_get_results(self, func, func_steps=10, results=None, steps=80):

        if not results:
            results = []

        activations = {}
        for neuron in self.neurons:
            activations[neuron] = []

        for i in range(steps):
            if func and  i % func_steps == 0:
                results.append(func(self.neurons))

                for neuron in self.neurons:
                    activations[neuron].append(neuron.get_activation())

            self.run()

        return results, activations
        
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

