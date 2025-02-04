
import os
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, path=""):
        self.output_path = path

    @classmethod
    def create_plotter(cls, path=None):
        return cls(path=path)

    def plot(self, history):

        # Plot loss
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training loss', 'val_loss'])
        plt.savefig(os.path.join(self.output_path, 'loss.png'))
        plt.close()

        # Plot Euclidean Distance
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['edit_dist'])
        plt.plot(history.history['val_edit_dist'])
        plt.legend(['Euclidean Distance', 'Validation Euclidean Distance'])
        plt.savefig(os.path.join(self.output_path, 'edit_distance.png'))
        plt.close()

        # Plot Combine Score 
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['combined_score'])
        plt.plot(history.history['val_combined_score'])
        plt.legend(['Combined Score', 'Validation Combined Score'])
        plt.savefig(os.path.join(self.output_path, 'combined_score.png'))
        plt.close()


