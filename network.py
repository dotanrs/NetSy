from . import neuron as n

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

    def run(self):
        for neuron in self.neurons:
            neuron.run()

    @property
    def neuron_list(self):
        return self.neurons

    def run_and_get_activations(self, neurons = None, steps = 80):

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
        
    def run_and_get_phases(self, neurons = None, steps = 80):

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

