
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

from BadData_AppC import DataObject, AppendixCFunction, Trainer, Overlap
from NegSamplingMath_Unlearn_AppC_FullDataSet import NegMLP, NegTrainer, ActPoly, accuracy, NegOverlap

def check_device():
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")
    return DEVICE


DEVICE = check_device()


class Yeeplot_modular:
    """Class for plotting modular function experiment results.

    data: single data points dictionary,list or tuple of data points to plot
    label: label or list of labels for the data
    x_label: label for x-axis
    y_label: label for y-axis
    latex_title: title in LaTeX format
    p: modulus
    title: title of the plot
    fontsize: font size for the plot"""

    def __init__(self, data, label, y_label, title, x_label="Training step", fontsize=10):
        self.data = data
        self.label = label
        self.x_label = x_label
        self.y_label = y_label
        self.fontsize = fontsize
        self.title = title

    def plot(self):
        plt.figure()
        try:
            if isinstance(self.data, list) and isinstance(self.label, list):
                try:
                    for d, l in zip(self.data, self.label):
                        plt.plot(d, label=l)
                except ValueError as ve:
                    print(f"ValueError during plotting: {ve}")
            else:
                plt.plot(self.data, label=self.label)
        except Exception as e:
            print(f"Error plotting data: {e}")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.legend()
        plt.show()

def setup_experiment_config(config: dict):

    modular_function = AppendixCFunction(config['c'], config['d'], config['p'])
    dataset = DataObject(modular_function, split=config['split'])

    model = NegMLP(config['p'], config['embedding_dim'], config['hidden'])
    model = model.to(DEVICE)

    trainer = NegTrainer(learning_rate=config['learning_rate'], num_negs_per_example=config['negs_per_ex'])
    
    return dataset, model, trainer

def train_model(trainer, model, dataset, config):
    trainer.train_model(
        model,
        dataset,
        max_steps=config['max_steps'],
        batch_size=config['batch_size'],
        weight_decay=config['weight_decay']
    )

def visualize_experiment(config: dict, dataset, model, animate: bool = False):
    latex_title = config['latex_title']
    p = config['p']

    # Losses
    Yeeplot_modular(
        data=[model.loss_dictionary['train_loss'], model.loss_dictionary['test_loss']],
        label=["Train Loss", "Test Loss"],
        y_label="Loss",
        title=f"Loss {latex_title} mod {p}" #I recommend to use f-strings for better readability, this is very powerful tool
    ).plot()

    # Accuracies
    Yeeplot_modular(
        data=[model.loss_dictionary['train_accuracy'], model.loss_dictionary['test_accuracy']],
        label=["Train Accuracy", "Test Accuracy"],
        y_label="Accuracy",
        title=f"Accuracy {latex_title} mod {p}"
    ).plot()

    # Positive-example accuracies
    Yeeplot_modular(
        data=[model.loss_dictionary['positive_train_accuracy'], model.loss_dictionary['positive_test_accuracy']],
        label=["Train Accuracy on positive examples", "Test Accuracy on positive examples"],
        y_label="Accuracy",
        title=f"Accuracy on positive examples {latex_title} mod {p}"
    ).plot()

    # Negative-example accuracies
    Yeeplot_modular(
        data=[model.loss_dictionary['negative_train_accuracy'], model.loss_dictionary['negative_test_accuracy']],
        label=["Train Accuracy on negative examples", "Test Accuracy on negative examples"],
        y_label="Accuracy",
        title=f"Accuracy on negative examples {latex_title} mod {p}"
    ).plot()

    #Do smth with it 
    if animate:
        # Animation for single-point test sets
        if len(dataset.test_data) == 1:

            rcParams['animation.embed_limit'] = 64  # MB, default is 20

            # Histogram
            example = dataset.test_data
            print(example)
            plt.figure()
            plt.bar(list(range(p)), model.loss_dictionary['counts_hist_all'])
            plt.title("Histogram " + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
            plt.xlabel("Question index")
            plt.ylabel("Count of predictions")
            plt.draw()
            plt.pause(0.001)
            plt.show()

            #Histogram with single prediction
            plt.figure()
            plt.bar(list(range(p)), model.loss_dictionary['prediction_mike'])
            plt.title("Histogram for single prediction " + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
            plt.xlabel("Question index")
            plt.ylabel("Count of predictions")
            plt.draw()
            plt.pause(0.001)
            plt.show()

            # Evolution of probability distribution
            probs = []
            for f in model.loss_dictionary.get('sorta_prob_dist', []):
                arr = f.detach().cpu().squeeze().numpy() if isinstance(f, torch.Tensor) else np.array(f).squeeze()
                s = float(arr.sum())
                if s > 0:  
                    arr = arr / s
                else:
                    print("Warning: Sum of probabilities is zero.")
                probs.append(arr)

            if probs:
                fig, ax = plt.subplots(figsize=(6,3))
                ax.set_title('Probability distribution evolution ' + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)   
                bars = ax.bar(range(len(probs[0])), probs[0])
                ax.set_ylim(0, 0.025) 
                txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)

                def update(i):
                    y = probs[i]
                    for b, h in zip(bars, y): 
                        b.set_height(float(h))
                    txt.set_text(f"step {i+1}/{len(probs)} | sum={y.sum():.3f} | argmax={int(np.argmax(y))}")
                    return bars
                
                anim = FuncAnimation(fig, update, frames=len(probs), interval=100, repeat=False)
                plt.close(fig)
                display(HTML(anim.to_jshtml()))
    

def run_and_visualize_experiment(config: dict):
    """
    Runs a single experiment based on a configuration dictionary and visualizes the results.
    """
    print("="*80)
    print(f"Starting Experiment: {config['latex_title']} (p={config['p']})")
    print("="*80)

    # 1. Setup

    dataset, model, trainer = setup_experiment_config(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    space_dim = (config['p'] ** 2) * 2**4
    pos_dim = config['p'] ** 2
    print(f"num_parameters: {num_params:,}; space_dim: {space_dim:,}; pos_dim: {pos_dim:,}")
    
    if config.get('print_test_len', False):
        print(len(dataset.test_data))

    # 2. Training
    train_model(trainer, model, dataset, config)
    
    # 3. Visualization 
    visualize_experiment(config, dataset, model)

    print("\n✅ Experiment Complete.\n")
