import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from netsy import network as n
from netsy import display as d
from netsy.factory import NeuronDict as nd
import random

# Parameters. Leave like this for a cool spiral effect
jee = 2.02
j = 1.2
thee = - 10 * (jee - j)
thei = - 10 * j

# Create a network with two sigmoid neurons. The definition of one as inihibitory
# and one as excitatory will be expressed in the connections
net = n.Network()
inhibitory = net.create_neuron(name="in", ntype=nd.sigmoid, bias=10, tanh_bias=thei, init=10.00001, log=False, der_step=0.002)
excitatory = net.create_neuron(name="ex", ntype=nd.sigmoid, bias=10, tanh_bias=thee, init=10.00001, log=False, der_step=0.002)

# Create connections. note that the excitatory neuron listens to itself
excitatory.listen_to(inhibitory, -j)
excitatory.listen_to(excitatory, jee)
inhibitory.listen_to(excitatory, j)

activations = net.run_and_get_activations(steps=50000)

# An example of changing the network in mid-run. comment to see without
# excitatory.listen_to(inhibitory, -2 * j)
# activations = net.run_and_get_activations(activations=activations, steps=100)

# Print the results

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

## left panel
ax1 = fig.add_subplot(121)
ax1.scatter(activations[:, inhibitory.index],activations[:, excitatory.index],color='blue', s=5, edgecolor='none')
plt.axis((9.9999, 10.0001, 9.9999, 10.0001))

# make axes square
ax1.set_aspect(1./ax1.get_data_ratio())

## right panel
plt.subplot(122)
d.show_all_in_graph([excitatory, inhibitory], activations, params="")

plt.show()