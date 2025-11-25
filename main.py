import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

from BadData_AppC import DataObject, AppendixCFunction, Trainer, Overlap
from NegSamplingMath_Unlearn_AppC_FullDataSet import NegMLP, NegTrainer, ActPoly, accuracy, NegOverlap



class Experiment:
    def __init__(self, config: dict): 
        self.experiment_parameters: dict = {} # Store experiment parameters
        self.device: torch.device = None # Device will be set up later

        checked_config = self.check_config(config)
        self.experiment_parameters.update(checked_config)

    def train_model(trainer, model, dataset, config):
        trainer.train_model(
            model,
            dataset,
            max_steps=config['max_steps'],
            batch_size=config['batch_size'],
            weight_decay=config['weight_decay']
        )

    def check_device():
        if torch.backends.mps.is_available():
            device = torch.device("mps")

        elif torch.cuda.is_available():
            device = torch.device("cuda")
            
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device
    
    def setup_experiment_config(self):

        self.device = self.check_device()

        modular_function = AppendixCFunction(self.experiment_parameters['c'], self.experiment_parameters['d'], self.experiment_parameters['p'])
        dataset = DataObject(modular_function, split=self.experiment_parameters['split'])

        model = NegMLP(self.experiment_parameters['p'], self.experiment_parameters['embedding_dim'], self.experiment_parameters['hidden'])
        
        if self.device is not None:
            model = model.to(self.device)
        else:
            model = model.to(torch.device("cpu")) # Fallback to CPU if no device is specified

        trainer = NegTrainer(learning_rate=self.experiment_parameters['learning_rate'], num_negs_per_example=self.experiment_parameters['negs_per_ex'])
        
        return dataset, model, trainer

        
    def check_config(self, config: dict) -> dict:
        required_keys = ('p', 
                         'c', 
                         'd', 
                         'embedding_dim', 
                         'hidden', 
                         'learning_rate',
                         'negs_per_ex', 
                         'max_steps', 
                         'batch_size', 
                         'weight_decay',
                         'split', 
                         'latex_title')
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        return config

    def calculate_parameters(self):
        num_params, memory_mb = self.calculat_model_size()
        # print(f"Model size: {memory_mb:.2f} MB")
        space_dim = (self.experiment_parameters['p'] ** 2) * 2**4
        pos_dim = self.experiment_parameters['p'] ** 2
        # print(f"num_parameters: {num_params:,}; space_dim: {space_dim:,}; pos_dim: {pos_dim:,}")
        self.experiment_parameters.update({
            'num_parameters': num_params,
            'memory_mb': memory_mb,
            'space_dim': space_dim,
            'pos_dim': pos_dim
        })

    def calculat_model_size(self)-> tuple[int,float]:
        num_params = sum(parameter.numel() for parameter in model.parameters()) # calculate total number of parameters of the model
        memory_bytes = num_params * 4  # 4 bytes for each float32 parameter
        memory_mb = memory_bytes / (1024 ** 2)
        return num_params,memory_mb

    @staticmethod
    def print_info(string: str):
        """Helper function to print information with formatting."""
        print("\n" + "="*80)
        print(string)
        print("="*80 + "\n")

    def run(self):
        """
        Runs a single experiment based on a configuration dictionary and visualizes the results.
        """
        self.print_info(f"Starting Experiment: {self.experiment_parameters['latex_title']} mod {self.experiment_parameters['p']}")

        dataset, model, trainer = self.setup_experiment_config()
        self.calculate_parameters()
        
        if self.experiment_parameters.get('print_test_len', False):
            print(len(dataset.test_data))

        self.train_model(trainer, model, self.dataset, self.experiment_parameters)
        self.visualize_experiment()

        self.print_info("Experiment Complete.")

    def plot_metics(self, labels, data, y_label, title, x_label="Training step"):
        plt.figure()
        try:
            if isinstance(data, list) and isinstance(labels, list):
                try:
                    for d, l in zip(data, labels):
                        plt.plot(d, label=l)
                except ValueError as ve:
                    print(f"ValueError during plotting: {ve}")
            else:
                plt.plot(data, label=labels)
        except Exception as e:
            print(f"Error plotting data: {e}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()

    def visualize_experiment(self, animate: bool = False):
        latex_title = self.experiment_parameters['latex_title']
        p = self.experiment_parameters['p']

        # Losses
        self.plot_metics(
            data=[self.model.loss_dictionary['train_loss'], self.model.loss_dictionary['test_loss']],
            label=["Train Loss", "Test Loss"],
            y_label="Loss",
            title=f"Loss {latex_title} mod {p}" #I recommend to use f-strings for better readability, this is very powerful tool
        ).plot()

        # Accuracies
        self.plot_metics(
            data=[self.model.loss_dictionary['train_accuracy'], self.model.loss_dictionary['test_accuracy']],
            label=["Train Accuracy", "Test Accuracy"],
            y_label="Accuracy",
            title=f"Accuracy {latex_title} mod {p}"
        ).plot()

        # Positive-example accuracies
        self.plot_metics(
            data=[self.model.loss_dictionary['positive_train_accuracy'], self.model.loss_dictionary['positive_test_accuracy']],
            label=["Train Accuracy on positive examples", "Test Accuracy on positive examples"],
            y_label="Accuracy",
            title=f"Accuracy on positive examples {latex_title} mod {p}"
        ).plot()

        # Negative-example accuracies
        self.plot_metics(
            data=[self.model.loss_dictionary['negative_train_accuracy'], self.model.loss_dictionary['negative_test_accuracy']],
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


