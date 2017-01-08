import matplotlib.pyplot as plt
from netsy import network as n
from netsy import display as d


### types:
# whitenoise
# threshold
# binarynoise

net = n.Network()

noise = net.create_neuron_array(name="noise", ntype="whitenoise", mean=0.1, size=7, lifespan=50, range=[-0.1, 1])

noise[1].set_lifespan(20)
noise[2].set_lifespan(70)
net.set_lifespan(noise[3:6], 45)

threshold = net.create_neuron_array(name="THRASH", ntype="threshold", size=20)
for neuron in threshold:
    neuron.listen_to(noise, "random")

net.set_lifespan(threshold[0:10], 20)
net.set_lifespan(threshold[10:14], 30)

top_level = net.create_neuron(name="top", ntype="threshold", log=True, threshold=2)
top_level.listen_to(threshold, 0.1)

activations = net.get_activations()

show_labels = False

plt.subplot(311)
d.show_all_in_graph(noise, activations, show_labels=show_labels)

plt.subplot(312)
d.show_all_in_graph(threshold, activations, show_labels=show_labels)

plt.subplot(313)
d.show_all_in_graph(top_level, activations, show_labels=True)

plt.show()