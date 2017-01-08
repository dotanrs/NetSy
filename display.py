import matplotlib.pyplot as plt

def show_all_in_graph(neurons, activations, show_labels=True):

    if not isinstance(neurons, list):
        neurons = [neurons]

    handles = []

    for i in neurons:
        handle, = plt.plot(activations[i], label=i.get_name())
        handles.append(handle)

    if show_labels:
        plt.legend(handles=handles)
    # plt.axis([0, 80, -1, 4])