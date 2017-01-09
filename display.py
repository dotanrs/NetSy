import matplotlib.pyplot as plt

def show_all_in_graph(neurons, activations, show_labels=True, params=""):

    if not isinstance(neurons, list):
        neurons = [neurons]

    handles = []

    for i in neurons:
        handle, = plt.plot(activations[i], params, label=i.get_name())
        handles.append(handle)

    if show_labels:
        plt.legend(handles=handles)
    # plt.axis([0, 80, -1, 4])