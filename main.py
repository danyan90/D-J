import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import torch
from typing import Tuple, List

from BadData_AppC import DataObject, AppendixCFunction
from NegSamplingMath_Unlearn_AppC_FullDataSet import NegMLP, NegTrainer

# ================================================================
# Config format:
# {
#     'p': int,                # Modulus
#     'c': List[int],          # Coefficients of polynomial
#     'd': List[int],          # Degrees of polynomial
#     'embedding_dim': int,    # Embedding dimension for model
#     'hidden': List[int],     # Hidden layer sizes for model
#     'learning_rate': float,  # Learning rate for trainer
#     'negs_per_ex': int,      # Number of negative samples per example
#     'max_steps': int,        # Maximum training steps
#     'batch_size': int,       # Batch size for training
#     'weight_decay': float,   # Weight decay for optimizer
#     'split': float,          # Train/test split ratio
#     'latex_title': str       # LaTeX formatted title for plots
# }
# ================================================================


class Experiment:
    # ================================================================
    # 1. INITIALIZATION & CONFIGURATION
    # ================================================================
    
    model: NegMLP
    trainer: NegTrainer
    dataset: DataObject
    device: torch.device

    def __init__(self, config: dict): 
        self.experiment_parameters: dict = {} # Store experiment parameters

        checked_config: dict = self.check_config(config)
        self.experiment_parameters.update(checked_config)

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
    
    # ================================================================
    # 2. MAIN EXPERIMENT METHODS
    # ================================================================
    
    def run(self, animate: bool = False) -> None:
        """Execute full experiment pipeline: setup, train, and visualize.
        
        Args:
            animate: Whether to create animated visualizations (only for single-example test sets), 
            default is False
        """
        self.print_info(
            f"Starting Experiment: {self.experiment_parameters['latex_title']} mod {self.experiment_parameters['p']}")

        self.dataset, self.model, self.trainer = self.setup_experiment_config()
        self.calculate_parameters()
        
        # if self.experiment_parameters.get('print_test_len', 
        #                                   False):
        #     print(len(self.dataset.test_data))
        self.train_model()
        self.visualize_experiment(animate=animate)

        self.print_info("Experiment Complete.")

    # ================================================================
    # 3. SETUP & PREPARATION
    # ================================================================
        
    def setup_experiment_config(self) -> Tuple[DataObject, 
                                               NegMLP, 
                                               NegTrainer]:
        """Setup dataset, model, and trainer for the experiment.
        
        Returns:
            Tuple of (dataset, model, trainer)
        """
        self.device = self.check_device()

        modular_function = AppendixCFunction(self.experiment_parameters['c'], 
                                             self.experiment_parameters['d'], 
                                             self.experiment_parameters['p'])
        dataset = DataObject(modular_function, 
                             split=self.experiment_parameters['split'])

        model = NegMLP(self.experiment_parameters['p'], 
                       self.experiment_parameters['embedding_dim'], 
                       self.experiment_parameters['hidden'])
        
        if self.device is not None:
            model = model.to(self.device)
        else:
            model = model.to(torch.device("cpu")) # Fallback to CPU if no device is specified

        trainer = NegTrainer(learning_rate=self.experiment_parameters['learning_rate'], 
                             num_negs_per_example=self.experiment_parameters['negs_per_ex'])
        
        return dataset, model, trainer
    
    def check_device(self) -> torch.device: # for Mac mps 
        if torch.backends.mps.is_available():
            device = torch.device("mps")

        elif torch.cuda.is_available():
            device = torch.device("cuda")
            
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device

    def calculate_parameters(self) -> None:
        """Calculate and store model statistics. Must be called after setup_experiment_config()."""
        if self.model is None:
            raise ValueError("Model must be initialized before calculating parameters.")
        num_params, memory_mb = self.calculate_model_size()
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

    def calculate_model_size(self) -> Tuple[int, float]:
        """Calculate model size in parameters and megabytes.
        
        Returns:
            Tuple of (num_parameters, memory_in_mb)
        """
        num_params = sum(parameter.numel() for parameter in self.model.parameters())
        memory_bytes = num_params * 4  # 4 bytes per float32
        memory_mb = memory_bytes / (1024 ** 2)  # Convert to MB
        return num_params, memory_mb
    
    # ================================================================
    # 4. TRAINING
    # ================================================================

    def train_model(self) -> None:
        """Train the model using configured trainer. Must be called after setup_experiment_config()."""
        # Confirm that trainer, model, and dataset are initialized
        if self.trainer is None or self.model is None or self.dataset is None:
            raise ValueError("Trainer, model, and dataset must be initialized before training.")
        
        # Train the model using the trainer
        self.trainer.train_model(
            self.model,
            self.dataset,
            max_steps=self.experiment_parameters['max_steps'],
            batch_size=self.experiment_parameters['batch_size'],
            weight_decay=self.experiment_parameters['weight_decay']
        )
    
    # ================================================================
    # 5. VISUALIZATION
    # ================================================================
    
    def visualize_experiment(self, animate: bool = False) -> None:
        """Visualize all experiment results. Must be called after train_model().
        
        Args:
            animate: Whether to create animated visualizations for single-example test sets
        """
        if not hasattr(self.model, 'loss_dictionary') or not self.model.loss_dictionary:
            raise ValueError("No training data to visualize. Call train_model() first.")
        
        # Plot all standard metrics
        self._plot_losses()
        self._plot_accuracies()
        self._plot_positive_accuracies()
        self._plot_negative_accuracies()
        
        # Plot animations if requested
        if animate:
            self._plot_animations()
   
    def plot_metrics(self, data: List | np.ndarray, labels: List[str] | str, 
                     y_label: str, title: str, x_label: str = "Training step") -> None:
        """Plot training metrics such as loss or accuracy.
         
        Args:
            data: Data series to plot (single array or list of arrays)
            labels: Labels for the data series (single string or list of strings)
            y_label: Label for the y-axis
            title: Title of the plot
            x_label: Label for the x-axis (default: "Training step")
        """
        plt.figure(figsize=(10, 6))
    
        if isinstance(data, list) and isinstance(labels, list):
            for d, label in zip(data, labels):
                plt.plot(d, label=label, linewidth=2)
        else:
            plt.plot(data, label=labels, linewidth=2)
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # --- Metrics plotting  ---

    def _plot_losses(self) -> None:
        """Plot training and test losses."""
        latex_title = self.experiment_parameters['latex_title']
        p = self.experiment_parameters['p']
        
        self.plot_metrics(
            data=[self.model.loss_dictionary['train_loss'], 
                  self.model.loss_dictionary['test_loss']],
            labels=["Train Loss", "Test Loss"],
            y_label="Loss",
            title=f"Loss: {latex_title} mod {p}"
        )
    
    def _plot_accuracies(self) -> None:
        """Plot overall training and test accuracies."""
        latex_title = self.experiment_parameters['latex_title']
        p = self.experiment_parameters['p']
        
        self.plot_metrics(
            data=[self.model.loss_dictionary['train_accuracy'], 
                  self.model.loss_dictionary['test_accuracy']],
            labels=["Train Accuracy", "Test Accuracy"],
            y_label="Accuracy",
            title=f"Accuracy: {latex_title} mod {p}"
        )
    
    def _plot_positive_accuracies(self) -> None:
        """Plot accuracies on positive examples only."""
        latex_title = self.experiment_parameters['latex_title']
        p = self.experiment_parameters['p']
        
        self.plot_metrics(
            data=[self.model.loss_dictionary['positive_train_accuracy'], 
                  self.model.loss_dictionary['positive_test_accuracy']],
            labels=["Train Accuracy (positive)", "Test Accuracy (positive)"],
            y_label="Accuracy",
            title=f"Positive examples: {latex_title} mod {p}"
        )
    
    def _plot_negative_accuracies(self) -> None:
        """Plot accuracies on negative examples only."""
        latex_title = self.experiment_parameters['latex_title']
        p = self.experiment_parameters['p']
        
        self.plot_metrics(
            data=[self.model.loss_dictionary['negative_train_accuracy'], 
                  self.model.loss_dictionary['negative_test_accuracy']],
            labels=["Train Accuracy (negative)", "Test Accuracy (negative)"],
            y_label="Accuracy",
            title=f"Negative examples: {latex_title} mod {p}"
        )
    
    def _plot_animations(self) -> None:
        """Create animated visualizations for single-example test sets."""
        if len(self.dataset.test_data) != 1:
            return  # Only animate for single-example test sets
        
        rcParams['animation.embed_limit'] = 64  # MB limit for animations
        
        # Create individual animation plots
        self._plot_histogram()
        self._plot_single_prediction_histogram()
        self._plot_probability_evolution()
    
    def _plot_histogram(self) -> None:
        """Plot histogram of all predictions."""
        plt.figure(figsize=(10, 5))
        plt.bar(range(self.experiment_parameters['p']), self.model.loss_dictionary['counts_hist_all'])
        plt.title(f"Histogram: {self.experiment_parameters['latex_title']} mod {self.experiment_parameters['p']} for {self.dataset.test_data.tolist()}", 
                  fontsize=10)
        plt.xlabel("Answer index")
        plt.ylabel("Prediction count")
        plt.tight_layout()
        plt.show()
    
    def _plot_single_prediction_histogram(self) -> None:
        """Plot histogram for single prediction."""
        plt.figure(figsize=(10, 5))
        plt.bar(range(self.experiment_parameters['p']), self.model.loss_dictionary['prediction_mike'])
        plt.title(f"Single prediction: {self.experiment_parameters['latex_title']} mod {self.experiment_parameters['p']} for {self.dataset.test_data.tolist()}", 
                  fontsize=10)
        plt.xlabel("Answer index")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
    
    def _plot_probability_evolution(self) -> None:
        """Create animated plot of probability distribution evolution."""
        # Prepare probability data
        probs = []
        for f in self.model.loss_dictionary.get('sorta_prob_dist', []):
            arr = f.detach().cpu().squeeze().numpy() if isinstance(f, torch.Tensor) else np.array(f).squeeze()
            s = float(arr.sum())
            if s > 0:  
                arr = arr / s
            else:
                print("Warning: Sum of probabilities is zero.")
            probs.append(arr)

        if not probs:
            return
        
        # Create animation
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"Probability evolution: {self.experiment_parameters['latex_title']} mod {self.experiment_parameters['p']} for {self.dataset.test_data.tolist()}", fontsize=10)   
        bars = ax.bar(range(len(probs[0])), probs[0])
        
        # Dynamic y-limit based on actual data
        max_prob = max(max(prob) for prob in probs)
        ax.set_ylim(0, max_prob * 1.2)
        
        txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(i: int):
            """Update function for animation frame."""
            y = probs[i]
            for b, h in zip(bars, y): 
                b.set_height(float(h))
            txt.set_text(f"Step {i+1}/{len(probs)} | sum={y.sum():.3f} | argmax={int(np.argmax(y))}")
            return bars
        
        anim = FuncAnimation(fig, update, frames=len(probs), interval=100, repeat=False)
        plt.close(fig)
        display(HTML(anim.to_jshtml()))

    # ================================================================
    # 6. UTILITIES
    # ================================================================
    
    @staticmethod
    def print_info(string: str):
        """Helper function to print information with formatting."""
        print("\n" + "="*80)
        print(string)
        print("="*80 + "\n")
