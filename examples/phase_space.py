import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from netsy import network as n
from netsy import display as d
from netsy.factory import NeuronDict as nd
import random

jee = 2.02
j = 1.2
thee = - 10 * (jee - j)
thei = - 10 * j

net = n.Network()

inhibitory = net.create_neuron(name="in", ntype=nd.sigmoid, bias=10, tanh_bias=thei, init=10.00001, log=False)
excitatory = net.create_neuron(name="ex", ntype=nd.sigmoid, bias=10, tanh_bias=thee, init=10.00001, log=False)


excitatory.listen_to(inhibitory, -j)
excitatory.listen_to(excitatory, jee)

inhibitory.listen_to(excitatory, j)

activations = net.run_and_get_activations(steps=100000)#00)

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(121)

# plt.show()

ax1.scatter(activations[inhibitory],activations[excitatory],color='blue',s=5,edgecolor='none')

N=1000
x = np.random.randn(N)
y = np.random.randn(N)

## left panel
# ax1.scatter(x,y,color='blue',s=5,edgecolor='none')
ax1.set_aspect(1./ax1.get_data_ratio()) # make axes square


plt.subplot(122)

d.show_all_in_graph([excitatory, inhibitory], activations, params="")


plt.show()