import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from netsy import network as n
from netsy import display as d
from netsy.factory import NeuronDict as nd
import random

# initiate the network
net = n.Network()

# add noies neurons to the network. The will not have synapses yet
noise = net.create_neuron_array(name="noise", ntype=nd.whitenoise, mean=0.1, size=8, lifespan=50, range=[-0.1, 1])

# To make things more interesting, kill them at different points
noise[1].set_lifespan(20)
noise[2].set_lifespan(70)
noise[3].set_lifespan(70)
noise[7].set_lifespan(70)
net.set_lifespan(noise[3:6], 45)

# add more neurons, this time threshold neurons
threshold = net.create_neuron_array(name="THRASH", ntype=nd.threshold, size=20)
net.set_lifespan(threshold[0:10], 20)
net.set_lifespan(threshold[10:14], 30)

# Create synapses. each threshold neuron will be connected to 
# (will receive input from) all the noise neurons
for neuron in threshold:
	# To create randomness, each connection will be of random strength
    neuron.listen_to(noise, "random")

# Create more neurons, this time sigmoid
sigmoid = net.create_neuron_array(name="sig", ntype=nd.sigmoid, size=3, der_step=0.1, tanh_bias=-2)

# Each one will be connected to a few noise neurons by random
for neuron in sigmoid:
	listento = [n for n in noise if random.random() > 0.7]
	neuron.listen_to(listento, 10)

# Select this neuron to print it's logs to console
# sigmoid[-1].show_log()

# create one more neuron that listenes to the last two layers.
# This can be thought of as the decision making neuron
top_level = net.create_neuron(name="top", ntype=nd.threshold, log=True, threshold=2)
top_level.listen_to(threshold, 10)
top_level.listen_to(sigmoid, 5)

# run the network and print the results
activations = net.run_and_get_activations()

show_labels = False

plt.subplot(411)
d.show_all_in_graph(noise, activations, show_labels=show_labels)
plt.title("noise neurons - layer 0")

plt.subplot(412)
d.show_all_in_graph(sigmoid, activations, show_labels=show_labels)
plt.title("sigmoid neurons - layer 1")

plt.subplot(413)
d.show_all_in_graph(threshold, activations, show_labels=show_labels)
plt.title("threshold neurons - layer 1")

plt.subplot(414)
d.show_all_in_graph(top_level, activations, show_labels=True)
plt.title("top_level neuron - layer 2")

plt.show()