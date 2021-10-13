import matplotlib.pyplot as plt
import numpy as np

def heatmap2d(arr: np.ndarray, cmap, output_file):
        plt.imshow(arr, cmap=cmap)
        plt.colorbar()
        plt.show()
        plt.savefig(output_file)

def PlotTrainingComp(nEpochs, train, val, ylabel, output_file):
        epochs = range(1, nEpochs + 1)
        plt.figure(0)
        plt.plot(epochs, train, color='black', label='Training set')
        plt.plot(epochs, val, 'b', label='Validation set')
        plt.title('')
        plt.ylim([0.5*np.min(val),1.5*np.min(val)])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(output_file)
