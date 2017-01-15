import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from netsy import network as n
from netsy import display as d
from netsy.factory import NeuronDict as nd
import random

NUM_PATTERNS = 3
PATTERN_PROB = 0.3
STEPS = 140
NUM_NEURONS = 150
DER_STEP = 0.01

# Create a network of sigmoid neurons
net = n.Network()
neurons = net.create_neuron_array(ntype=nd.sigmoid, size=NUM_NEURONS, init=0, der_step=DER_STEP)



# create all-to-all connectivity
net.all_to_all_connectivity(neurons, 0)

# create connection strength according to the hopfield formula
connection_strength = 1 / NUM_PATTERNS
patterns = []
for i in range(NUM_PATTERNS):
	active = [n for n in neurons if random.random() < PATTERN_PROB]
	others = [n for n in neurons if n not in active]
	patterns.append(active)

	for n in neurons:
		if n in active:
			q = 1
		else:
			q = -1

		n.increase_connection_strength(active, connection_strength * q)
		n.increase_connection_strength(others, connection_strength * -q)



# start the network at a pattern and check the activity.
# is the pattern activity stable?
# does it converge to one of the patterns?
import matplotlib.pyplot as plt
import numpy as np

pattern = patterns[-1]

# set the initial state of the network. You can start the network from
# a pattern and add noise, or just use noise to see if a pattern is selected

# net.apply_pattern(pattern)
# the parameter of the noise function is the size of the scatter
net.apply_noise(0.5)

# helper function that check if the pattern matches the activity
def calc_distance_from_pattern(pattern, neurons):
	active = [n for n in neurons if n.get_activation() > 0]
	wrong = len([n for n in active if n not in pattern]) + \
			len([n for n in pattern if n not in active])
	score = 1 - wrong / len(neurons)
	return score

def calc_distance_from_all_patterns(neurons):
	scores = [0] * len(patterns)
	for i, p in enumerate(patterns):
		scores[i] = calc_distance_from_pattern(p, neurons)

	return scores

# first plot: the similarity of the activity and the patterns.
# Does it converge to one pattern?
plt.subplot(211)
results, activations = net.run_and_get_results(func=calc_distance_from_all_patterns, steps=STEPS)
handles = []
for i, p in enumerate(patterns):
	res = [x[i] for x in results]
	handle, = plt.plot(res, "-o", label=i)
	handles.append(handle)

plt.legend(handles=handles)

# second plot: the activity of the neurons. Is it (eventually) stable?
plt.subplot(212)
d.show_all_in_graph(neurons, activations, show_labels=False)
plt.show()
