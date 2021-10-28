import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


markers = ('s', 'x', 'o', '^', 'v', '+', 'x')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'darkorange', 'mediumpurple')


def plot_data(data, label, resolution=0.02, names='default', ax=plt):
    cmap = ListedColormap(colors[:len(np.unique(label))])

    if names == 'default' or not isinstance(names, dict):
        names = {lbl: lbl for lbl in label}
    else:
        for lbl in label:
            if lbl not in names.keys():
                names[lbl] = lbl

    # Defining axis
    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Draw scattrered data
    for idx, cl in enumerate(np.unique(label)):
        ax.scatter(x=data[label == cl, 0], y=data[label == cl, 1],
                   alpha=0.8, c=colors[idx], marker=markers[idx], label=names[cl], edgecolor='black')


def plot_decision_regions(data, label, classifier, resolution=0.02, names='default', ax=plt):
    if names == 'default' or not isinstance(names, dict):
        names = {lbl:lbl for lbl in label}
    else:
        for lbl in label:
            if lbl not in names.keys():
                names[lbl] = lbl

    # Defining axis
    x1_min, x1_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    x2_min, x2_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # Predict label
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    if Z.ndim == 2:
        Z = np.array([d.argmax() for d in Z])
    Z = Z.reshape(xx1.shape)

    cmap = ListedColormap(colors[:len(np.unique(Z))])

    # Draw contour line
    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    # Draw scattrered data
    for idx, cl in enumerate(np.unique(label)):
        ax.scatter(x=data[label == cl, 0], y=data[label == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], label=names[cl], edgecolor='black')