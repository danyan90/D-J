import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML, Markdown
import torch
from typing import Tuple, List

from BadData_AppC import DataObject
from NegSamplingMath_Unlearn_AppC_FullDataSet import NegMLP, NegTrainer

import time
from functools import wraps

def timing(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start
        # elapsed_in_ms = elapsed * 1000
        print(f"{func.__name__:30s} executed in {elapsed:10.2f} s")
        # print(f"{func.__name__:30s} executed in {elapsed_in_ms:10.2f} ms")
        return result
    return wrapper

# ================================================================
# Config format:
# {
#     'p': int,                # Modulus
#     'c': List[int],          # Coefficients of polynomial
#     'd': List[int],          # Degrees of polynomial
#     'embedding_dim': int,    # Embedding dimension for model
#     'hidden': int,           # Hidden layer size for model
#     'learning_rate': float,  # Learning rate for trainer
#     'negs_per_ex': int,      # Number of negative samples per example
#     'max_steps': int,        # Maximum training steps
#     'batch_size': int,       # Batch size for training
#     'weight_decay': float,   # Weight decay for optimizer
#     'split': float,          # Train/test split ratio
#     'latex_title': str       # LaTeX formatted title for plots
# }
# ================================================================

class ModularFunction:
    #Here we define a modular polynomial of the form 
    # (c1*x^d1 + c2*x^d2)^d3 + c3*x^d4*x^d5. 
    # So we take c = [c1 c2 c3] and d = [d1 d2 d3 d4 d5]
    
    def __init__(self, c: int, d: int, p: int):
        self.c = c
        self.d = d
        self.p = p

    def __call__(self, x: int, y: int):
        c1, c2, c3 = self.c
        d1, d2, d3, d4, d5 = self.d
        return ((c1*(x**d1) + c2*(y**d2))**d3 + c3*(x**d4)*(y**d5)) % self.p
    
    @property
    def latex_title(self) -> str:
        """Generate LaTeX representation from coefficients automatically."""
        c1, c2, c3 = self.c
        d1, d2, d3, d4, d5 = self.d
        
        if c1 == 0:
            x_part = ""
        elif c1 == 1 and d1 == 1:
            x_part = "x"
        elif c1 == 1:
            x_part = f"x^{{{d1}}}"
        elif d1 == 1:
            x_part = f"{c1}x"
        else:
            x_part = f"{c1}x^{{{d1}}}"

        if c2 == 0:
            y_part = ""
        elif c2 == 1 and d2 == 1:
            y_part = "y"
        elif c2 == 1:
            y_part = f"y^{{{d2}}}"
        elif d2 == 1:
            y_part = f"{c2}y"
        else:
            y_part = f"{c2}y^{{{d2}}}"

        # Build the sum part with proper sign handling
        if x_part and y_part:
            operator = " " if str(c2).startswith('-') else " + "
            sum_part = f"{x_part}{operator}{y_part}"
        elif x_part:
            sum_part = x_part
        elif y_part:
            sum_part = y_part
        else:
            sum_part = "0"

        # Build outer expression
        if d3 == 0:
            outer = "1"
        elif d3 == 1:
            outer = sum_part
        else:
            outer = f"({sum_part})^{{{d3}}}"

        # Build additional term with proper sign handling
        if c3 != 0:
            # Build the c3 part (with coefficient)
            if c3 == 1:
                c3_str = ""  # Omit coefficient 1
            elif c3 == -1:
                c3_str = "-"  # Just the minus sign
            else:
                c3_str = str(c3)
            
            # Build the xy part
            if d4 == 0 and d5 == 0:
                xy_part = "1"
            elif d4 == 0:
                xy_part = f"y^{{{d5}}}" if d5 != 1 else "y"
            elif d5 == 0:
                xy_part = f"x^{{{d4}}}" if d4 != 1 else "x"
            elif d4 == 1 and d5 == 1:
                xy_part = "xy"
            elif d4 == 1:
                xy_part = f"xy^{{{d5}}}"
            elif d5 == 1:
                xy_part = f"x^{{{d4}}}y"
            else:
                xy_part = f"x^{{{d4}}}y^{{{d5}}}"
            
            # Combine with sign
            sign = " + " if c3 > 0 else " "
            term = f"{sign}{c3_str}{xy_part}"
        else:
            term = ""
        
        return f"$f(x, y) = {outer}{term} \\,\\, mod \\,\\, {self.p}$"



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
    
    @timing
    def run(self, animate: bool = False) -> None:
        """Execute full experiment pipeline: setup, train, and visualize.
        
        Args:
            animate: Whether to create animated visualizations (only for single-example test sets), 
            default is False
        """
        self.print_info(
            f"Starting Experiment: {self.latex_title} mod {self.p}")

        self.dataset, self.model, self.trainer = self.setup_experiment_config()
        self.calculate_parameters()

        print(f"\nModel: {self.experiment_parameters['num_parameters']:,} parameters")
        print(f"Memory: {self.experiment_parameters['memory_mb']:.2f} MB")
        print(f"Space: {self.experiment_parameters['space_dim']:,} dimensions\n")
        
        
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

        modular_function = ModularFunction(self.c, 
                                             self.d, 
                                             self.p)
        dataset = DataObject(modular_function, 
                             split=self.split)

        model = NegMLP(self.p, 
                       self.embedding_dim, 
                       self.hidden)
        
        if self.device is not None:
            model = model.to(self.device)
        else:
            model = model.to(torch.device("cpu")) # Fallback to CPU if no device is specified

        trainer = NegTrainer(learning_rate=self.learning_rate, 
                             num_negs_per_example=self.negs_per_ex)
        
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
        space_dim = (self.p ** 2) * 2**4
        pos_dim = self.p ** 2
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
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay
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
        latex_title = self.latex_title
        p = self.p
        
        self.plot_metrics(
            data=[self.model.loss_dictionary['train_loss'], 
                  self.model.loss_dictionary['test_loss']],
            labels=["Train Loss", "Test Loss"],
            y_label="Loss",
            title=f"Loss: {latex_title} mod {p}"
        )
    
    def _plot_accuracies(self) -> None:
        """Plot overall training and test accuracies."""
        latex_title = self.latex_title
        p = self.p
        
        self.plot_metrics(
            data=[self.model.loss_dictionary['train_accuracy'], 
                  self.model.loss_dictionary['test_accuracy']],
            labels=["Train Accuracy", "Test Accuracy"],
            y_label="Accuracy",
            title=f"Accuracy: {latex_title} mod {p}"
        )
    
    def _plot_positive_accuracies(self) -> None:
        """Plot accuracies on positive examples only."""
        latex_title = self.latex_title
        p = self.p
        
        self.plot_metrics(
            data=[self.model.loss_dictionary['positive_train_accuracy'], 
                  self.model.loss_dictionary['positive_test_accuracy']],
            labels=["Train Accuracy (positive)", "Test Accuracy (positive)"],
            y_label="Accuracy",
            title=f"Positive examples: {latex_title} mod {p}"
        )
    
    def _plot_negative_accuracies(self) -> None:
        """Plot accuracies on negative examples only."""
        latex_title = self.latex_title
        p = self.p
        
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
        plt.bar(range(self.p), self.model.loss_dictionary['counts_hist_all'])
        plt.title(f"Histogram: {self.latex_title} mod {self.p} for {self.dataset.test_data.tolist()}", 
                  fontsize=10)
        plt.xlabel("Answer index")
        plt.ylabel("Prediction count")
        plt.tight_layout()
        plt.show()
    
    def _plot_single_prediction_histogram(self) -> None:
        """Plot histogram for single prediction."""
        plt.figure(figsize=(10, 5))
        plt.bar(range(self.p), self.model.loss_dictionary['prediction_mike'])
        plt.title(f"Single prediction: {self.latex_title} mod {self.p} for {self.dataset.test_data.tolist()}", 
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
        ax.set_title(f"Probability evolution: {self.latex_title} mod {self.p} for {self.dataset.test_data.tolist()}", fontsize=10)   
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
        try:
            # Use display for Jupyter notebooks
            display(Markdown(string))
        except NameError:
            # Fallback to print for regular Python
            print("\n" + "="*80)
            print(string)
            print("="*80 + "\n")

    def __repr__(self):
        return f"Experiment({self.experiment_parameters})"
    
    # ================================================================
    # PROPERTIES FOR CONFIG PARAMETERS
    # ================================================================
    
    @property
    def p(self) -> int:
        """Get modulus p."""
        return self.experiment_parameters['p']
    
    @property
    def c(self) -> List[int]:
        """Get coefficients c."""
        return self.experiment_parameters['c']
    
    @property
    def d(self) -> List[int]:
        """Get coefficients d."""
        return self.experiment_parameters['d']
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.experiment_parameters['embedding_dim']
    
    @property
    def hidden(self) -> int:
        """Get hidden layer size."""
        return self.experiment_parameters['hidden']
    
    @property
    def learning_rate(self) -> float:
        """Get learning rate."""
        return self.experiment_parameters['learning_rate']
    
    @property
    def negs_per_ex(self) -> int:
        """Get number of negative samples per example."""
        return self.experiment_parameters['negs_per_ex']
    
    @property
    def max_steps(self) -> int:
        """Get maximum training steps."""
        return self.experiment_parameters['max_steps']
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.experiment_parameters['batch_size']
    
    @property
    def weight_decay(self) -> float:
        """Get weight decay."""
        return self.experiment_parameters['weight_decay']
    
    @property
    def split(self) -> float:
        """Get train/test split ratio."""
        return self.experiment_parameters['split']
    
    @property
    def latex_title(self) -> str:
        """Get LaTeX formatted title."""
        return self.experiment_parameters['latex_title']

if __name__ == "__main__":
    # Run example experiment


    config = {
        "p": 17, 
        "c": [4, 1, 0], 
        "d": [1, 2, 3, 0, 0], 
        "max_steps": 50,              
        "learning_rate": 0.005,
        "batch_size": 128,            
        "weight_decay": 1e-4,
        "embedding_dim": 2**5,        
        "hidden": 128,               
        "split": 0.99999,
        "negs_per_ex": 5              
        }

    
    experiment = Experiment(config)
    experiment.run(animate=False)