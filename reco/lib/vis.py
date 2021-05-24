def heatmap2d(arr: np.ndarray, cmap, output_file):
        plt.imshow(arr, cmap=cmap)
        plt.colorbar()
        plt.show()
        plt.savefig(output_file)
