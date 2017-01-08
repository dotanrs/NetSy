import matplotlib.pyplot as plt
from netsy import network as n
from netsy import display as d


net = n.Network()

noise = net.create_neuron_array(name="noise", ntype="binarynoise", size=10, mean=0.1, lifespan=50)

noise[1].set_lifespan(60)
noise[2].set_lifespan(70)
net.set_lifespan(noise[3:6], 65)

threshold = net.create_neuron_array(name="THRASH", ntype="threshold", size=20)
for neuron in threshold:
    neuron.listen_to(noise, "random")

top_level = net.create_neuron(name="top", ntype="threshold", log=True)
top_level.listen_to(threshold, 1)

activations = net.get_activations()

show_labels = False

plt.subplot(311)
d.show_all_in_graph(noise, activations, show_labels=show_labels)

plt.subplot(312)
d.show_all_in_graph(threshold, activations, show_labels=show_labels)

plt.subplot(313)
d.show_all_in_graph(top_level, activations, show_labels=show_labels)

plt.show()