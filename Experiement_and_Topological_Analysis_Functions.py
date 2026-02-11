#Here we use negative sampling to train the model. We also use t-SNE and persistent homology to analyze the embeddings.

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import imageio_ffmpeg
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
rcParams['animation.embed_limit'] = 1000  # MB, default is 20
from sklearn.manifold import TSNE
from ripser import ripser
from persim import plot_diagrams

from BadData_AppC import DataObject, AppendixCFunction, Classification, Trainer
from NegSamplingMath_Unlearn_AppC_FullDataSet import NegMLP, NegTrainer


# ====================================
# Here is the experiment where we output the loss dictionary
# ==================================== 

class RunTrainingExperiment:

    def __init__(self, config: dict):
        self.config = config

    def run_and_visualize_experiment_Classification(self):
        """
        Runs a single experiment based on a configuration dictionary and visualizes the results.
        """
        print("="*80)
        print(f"Starting Experiment: {self.config['latex_title']} (p={self.config['p']})")
        print("="*80)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")

        # 1. Setup
        modular_function = AppendixCFunction(self.config['c'], self.config['d'], self.config['p'])
        dataset = DataObject(modular_function, split=self.config['split'])
        model = Classification(self.config['p'], self.config['embedding_dim'], self.config['hidden'])
        model = model.to(DEVICE)
        trainer = Trainer(learning_rate=self.config['learning_rate'])
        
        num_params = sum(p.numel() for p in model.parameters())
        space_dim = (self.config['p'] ** 2) * 2**4
        pos_dim = self.config['p'] ** 2
        print(f"num_parameters: {num_params:,}; space_dim: {space_dim:,}; pos_dim: {pos_dim:,}")
        
        if self.config.get('print_test_len', False):
            print(len(dataset.test_data))

        # 2. Training
        trainer.train_model(
            model,
            dataset,
            max_steps=self.config['max_steps'],
            batch_size=self.config['batch_size'],
            weight_decay=self.config['weight_decay']
        )

        # Polynomial as a string
        latex_title = self.config['latex_title']
        p = self.config['p']
        SaveTorF = self.config['SaveTorF']
        poly_name = self.config['poly_name']

        # 3. Visualization
        # Plot Loss
        plt.figure()
        plt.plot(model.loss_dictionary['train_loss'], label="Train Loss")
        plt.plot(model.loss_dictionary['test_loss'], label="Test Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        if len(dataset.test_data) == 1:
            plt.title('Loss ' + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
        else:
            plt.title('Loss ' + latex_title + ' mod ' + str(p), fontsize=10)
        plt.legend()
        if len(dataset.test_data) ==1 and SaveTorF == True:
            plt.savefig('Loss ' + poly_name + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), dpi=300, bbox_inches='tight')
        elif SaveTorF == True:
            plt.savefig('Loss ' + poly_name + ' mod ' + str(p), dpi=300, bbox_inches='tight')
        plt.show()

        # Plot Accuracy
        plt.figure()
        plt.plot(model.loss_dictionary['train_accuracy'], label="Train Accuracy")
        plt.plot(model.loss_dictionary['test_accuracy'], label="Test Accuracy")
        plt.xlabel("Training step")
        plt.ylabel("Accuracy")
        if len(dataset.test_data) == 1:
            plt.title('Accuracy ' + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
        else:
            plt.title('Accuracy ' + latex_title + ' mod ' + str(p), fontsize=10)
        plt.legend()
        if len(dataset.test_data) ==1 and SaveTorF == True:
            plt.savefig('Accuracy ' + poly_name + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), dpi=300, bbox_inches='tight')
        elif SaveTorF == True:
            plt.savefig('Accuracy ' + poly_name + ' mod ' + str(p), dpi=300, bbox_inches='tight')   
        plt.show()

        # Animation for single-point test sets
        if len(dataset.test_data) == 1:

            rcParams['animation.embed_limit'] = 64  # MB, default is 20

            # Histogram
            example = dataset.test_data
            print(example)
            plt.figure()
            plt.bar(list(range(p)), model.loss_dictionary['counts_hist'])
            plt.title("Histogram " + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
            plt.xlabel("Question index")
            plt.ylabel("Count of predictions")

            # Evolution of probability distribution
            probs = []
            for f in model.loss_dictionary.get('prob_dist', []):
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
                ax.set_ylim(0, 1) 
                txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)

                def update(i):
                    y = probs[i]
                    for b, h in zip(bars, y): 
                        b.set_height(float(h))
                    txt.set_text(f"step {i+1}/{len(probs)} | sum={y.sum():.3f} | argmax={int(np.argmax(y))}")
                    return bars
                
                anim = FuncAnimation(fig, update, frames=len(probs), interval=100, repeat=False)
                if SaveTorF == True:
                    anim.save('probability_evolution_' + poly_name + '_mod_' + str(p) + '.mp4', writer="ffmpeg", dpi=150, fps=25)
                plt.close(fig)
                display(HTML(anim.to_jshtml()))
        
        print("\n✅ Experiment Complete.\n")
        return model.loss_dictionary

    def run_and_visualize_experiment_NegativeSampling(self):

        warnings.filterwarnings(
        "ignore",
        message="The input point cloud has more columns than rows; did you mean to transpose?"
        )

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")
    
        """
        Runs a single experiment based on a configuration dictionary and visualizes the results.
        """
        print("="*80)
        print(f"Starting Experiment: {self.config['latex_title']} (p={self.config['p']})")
        print("="*80)

        # 1. Setup
        modular_function = AppendixCFunction(self.config['c'], self.config['d'], self.config['p'])
        dataset = DataObject(modular_function, split=self.config['split'])
        model = NegMLP(self.config['p'], self.config['embedding_dim'], self.config['hidden'])
        model = model.to(DEVICE)
        trainer = NegTrainer(learning_rate=self.config['learning_rate'], num_negs_per_example=self.config['negs_per_ex'])
        
        num_params = sum(p.numel() for p in model.parameters())
        space_dim = (self.config['p'] ** 2) * 2**4
        pos_dim = self.config['p'] ** 2
        print(f"num_parameters: {num_params:,}; space_dim: {space_dim:,}; pos_dim: {pos_dim:,}")
        
        if self.config.get('print_test_len', False):
            print(len(dataset.test_data))

        # 2. Training
        trainer.train_model(
            model,
            dataset,
            max_steps = self.config['max_steps'],
            batch_size = self.config['batch_size'],
            weight_decay = self.config['weight_decay']
        )

        # Polynomial as a string
        latex_title = self.config['latex_title']
        p = self.config['p']
        poly_name = self.config['poly_name']
        SaveTorF = self.config['SaveTorF']

        # 3. Visualization
        #PLot losses
        plt.figure()
        plt.plot(model.loss_dictionary['train_loss'], label = "Train Loss")
        plt.plot(model.loss_dictionary['test_loss'], label="Test Loss") 
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        if len(dataset.test_data) == 1:
            plt.title('Loss ' + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
        else:
            plt.title('Loss ' + latex_title + ' mod ' + str(p), fontsize=10)
        plt.legend()
        if len(dataset.test_data) ==1 and SaveTorF == True:
            plt.savefig('Loss ' + poly_name + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), dpi=300, bbox_inches='tight')
        elif SaveTorF == True:
            plt.savefig('Loss ' + poly_name + ' mod ' + str(p), dpi=300, bbox_inches='tight')
        plt.show

        #Plot Accuracies
        plt.figure()
        plt.plot(model.loss_dictionary['train_accuracy'], label="Train Accuracy")
        plt.plot(model.loss_dictionary['test_accuracy'], label="Test Accuracy")
        plt.xlabel("Training step")
        plt.ylabel("Accuracy")
        if len(dataset.test_data) == 1:
            plt.title('Accuracy ' + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
        else:
            plt.title('Accuracy ' + latex_title + ' mod ' + str(p), fontsize=10)
        plt.legend()
        if len(dataset.test_data) ==1 and SaveTorF == True:
            plt.savefig('Accuracy ' + poly_name + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), dpi=300, bbox_inches='tight')
        elif SaveTorF == True:
            plt.savefig('Accuracy ' + poly_name + ' mod ' + str(p), dpi=300, bbox_inches='tight')   
        plt.show()

        #Plot Positive accuracies
        plt.figure()
        plt.plot(model.loss_dictionary['positive_train_accuracy'], label="Train Accuracy on positive examples")
        plt.plot(model.loss_dictionary['positive_test_accuracy'], label="Test Accuracy on positive examples")
        plt.xlabel("Training step")
        plt.ylabel("Accuracy")
        if len(dataset.test_data) == 1:
            plt.title('Accuracy on positive example ' + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
        else:
            plt.title('Accuracy on positive examples ' + latex_title + ' mod ' + str(p), fontsize=10)
        plt.legend()
        if len(dataset.test_data) ==1 and SaveTorF == True:
            plt.savefig('Accuracy on positive example ' + poly_name + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), dpi=300, bbox_inches='tight')
        elif SaveTorF == True:
            plt.savefig('Accuracy on positive examples ' + poly_name + ' mod ' + str(p), dpi=300, bbox_inches='tight')  
        plt.show()

        #Plot Negative accuracies
        plt.figure()   
        plt.plot(model.loss_dictionary['negative_train_accuracy'], label="Train Accuracy on negative examples")
        plt.plot(model.loss_dictionary['negative_test_accuracy'], label="Test Accuracy on negative examples")
        plt.xlabel("Training step")
        plt.ylabel("Accuracy")
        if len(dataset.test_data) == 1:
            plt.title('Accuracy on negative example ' + latex_title + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), fontsize=10)
        else:
            plt.title('Accuracy on negative examples ' + latex_title + ' mod ' + str(p), fontsize=10)
        plt.legend()
        if len(dataset.test_data) ==1 and SaveTorF == True:
            plt.savefig('Accuracy on negative examples ' + poly_name + ' mod ' + str(p) + ' for ' + str(dataset.test_data.tolist()), dpi=300, bbox_inches='tight')
        elif SaveTorF == True:
            plt.savefig('Accuracy on negative examples ' + poly_name + ' mod ' + str(p), dpi=300, bbox_inches='tight')  
        plt.show()

        #Return the loss dictionary 
        return model.loss_dictionary
        


# =================================
# Here is the t-SNE stuff
# =================================
class tSNEstuff:
    def __init__(self, config: dict):
        self.tSNE_step   = config['tSNE_step']
        self.sub_sample  = config['sub_sample']
        self.SaveTorF    = config['SaveTorF']
        self.latex_title = config['latex_title']
        self.poly_name   = config['poly_name']
        self.p           = config['p']

    def Run_tSNE(self, representation_type: str, point_clouds, Columns = False):
        seed = 42
        rng = np.random.default_rng(seed)

        embs = []
        tsne_steps = []
        fixed_idx = None

        for i, C in enumerate(point_clouds):
            step = i + 1
            if step != 1 and step % self.tSNE_step != 0:
                continue

            arr = C.detach().cpu().numpy() if isinstance(C, torch.Tensor) else np.array(C)
            arr = arr.reshape(arr.shape[0], -1)   # (N, d)

            if Columns:        # <- ADD THIS
                arr = arr.T    # now columns become points

            if fixed_idx is None:
                N = arr.shape[0]
                if self.sub_sample is None or N <= self.sub_sample:
                    fixed_idx = np.arange(N)
                else:
                    fixed_idx = rng.choice(N, size=self.sub_sample, replace=False)

            arr = arr[fixed_idx]
            embs.append(arr)
            tsne_steps.append(step)

        all_E = np.concatenate(embs, axis=0)   # (frames * m, d)

        frames = len(embs)
        m = embs[0].shape[0]

        tsne = TSNE(
            n_components=2,
            init="pca",
            perplexity=min(30, m - 1),
            learning_rate="auto",
            random_state=seed,
        )
        all_2d = tsne.fit_transform(all_E)
        frames_2d = all_2d.reshape(frames, m, 2)

        fig, ax = plt.subplots(figsize=(8, 6))
        scat = ax.scatter(frames_2d[0][:, 0], frames_2d[0][:, 1], s=10)
        ax.set_xlim(all_2d[:, 0].min(), all_2d[:, 0].max())
        ax.set_ylim(all_2d[:, 1].min(), all_2d[:, 1].max())

        def update(i):
            scat.set_offsets(frames_2d[i])
            ax.set_title(
                f"t-SNE evolution {representation_type} {self.latex_title} mod {self.p} "
                f"(step {tsne_steps[i]}/{tsne_steps[-1]})"
            )
            return scat,

        anim = FuncAnimation(fig, update, frames=frames, interval=120, repeat=False)

        if self.SaveTorF:
            anim.save(
                f"tsne_{representation_type}_{self.poly_name}_mod_{self.p}.mp4",
                writer="ffmpeg", dpi=150, fps=25
            )

        plt.close(fig)
        display(HTML(anim.to_jshtml()))


# ===================================================
# Here is the Persistent Homology stuff
# ====================================================

class PersistentHomology:

    def __init__(self, config: dict):
        self.subsample = config['sub_sample']
        self.PH_step =  config['PH_step']
        self.PrintPD_step = config['PrintPD_step']

    warnings.filterwarnings(
    "ignore",
    message="The input point cloud has more columns than rows; did you mean to transpose?"
    )

    @staticmethod
    def persistent_entropy(diagram: np.ndarray) -> float:
        """
        Compute persistent entropy for a single persistence diagram (one homology dimension).
        diagram: array of shape (n_bars, 2) with [birth, death].
        """
        if diagram.size == 0:
            return 0.0

        # Keep only finite deaths
        finite = diagram[np.isfinite(diagram[:, 1])]
        if finite.size == 0:
            return 0.0

        lengths = finite[:, 1] - finite[:, 0]
        lengths = lengths[lengths > 0]  # ignore zero-length bars

        if len(lengths) == 0:
            return 0.0

        L = lengths.sum()
        p = lengths / L
        return float(-(p * np.log(p)).sum())

    @staticmethod
    def barcode_plot(diagrams, ax=None):
        """
        Minimal barcode plotter that works regardless of persim version.
        diagrams: list of dgms: [H0_dgm, H1_dgm, H2_dgm, ...]
        """

        if ax is None:
            ax = plt.gca()

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        y = 0

        # Set some default x-limits in case we need to draw "infinite" bars
        xmin, xmax = np.inf, -np.inf
        for dgm in diagrams:
            if len(dgm) == 0:
                continue
            births = dgm[:, 0]
            deaths = dgm[:, 1]
            finite_deaths = deaths[np.isfinite(deaths)]
            if len(finite_deaths) > 0:
                xmin = min(xmin, births.min())
                xmax = max(xmax, finite_deaths.max())
        if not np.isfinite(xmin):
            xmin, xmax = 0.0, 1.0
        ax.set_xlim(xmin, xmax * 1.05)

        for dim, dgm in enumerate(diagrams):
            color = colors[dim % len(colors)]
            for birth, death in dgm:
                if not np.isfinite(death):
                    death = xmax * 1.02  # draw infinite bars slightly beyond range
                ax.hlines(y, birth, death, color=color, linewidth=2)
                y += 1

        ax.set_xlabel("Filtration")
        ax.set_ylabel("Bars (stacked)")
        ax.set_title("Barcodes (H0,H1,H2)")
        ax.grid(True)



    def Homology(self, representation_type: str, point_clouds, Columns = False):    
        # Here we do topological data analysis with perisstent homology on the context vectors
        # How often to analyze and how often to *plot* diagrams/barcodes

        H0_entropy = []
        H1_entropy = []
        H2_entropy = []
        topo_steps = []

        idx_sub = None

        for idx, E in enumerate(point_clouds):
            step = idx + 1  # 1-based training step index

            # Only analyze every PH_step-th snapshot and the first one
            if step != 1 and step % self.PH_step != 0:
                continue

            # Convert to numpy
            if isinstance(E, torch.Tensor):
                X = E.detach().cpu().numpy()
            else:
                X = np.array(E)

            if Columns:        # <- ADD THIS
                X = X.T    # now columns become points

            # ---- subsample vectors ----
            n_total = X.shape[0]
            n_sub   = min(self.subsample, n_total)
            if idx_sub is None:
                idx_sub = np.random.choice(n_total, size=n_sub, replace=False)
            X = X[idx_sub]

            #t-SNE at each step to help visualize
            tsne = TSNE(
                n_components=2,
                init="pca",
                perplexity=max(5, min(30, (X.shape[0] - 1) // 3)),
                learning_rate="auto",
                random_state=0,
            )
            proj_2d = tsne.fit_transform(X)

            # Ripser on point cloud X, up to H2
            #if X.shape[1] > X.shape[0]:
                #X = X.T
            res = ripser(X, maxdim=2)
            dgms = res["dgms"]  # list: dgms[0]=H0, dgms[1]=H1, dgms[2]=H2 (if present)

            topo_steps.append(step)

            # H0 entropy (always present)
            H0_entropy.append(self.persistent_entropy(dgms[0]))

            # H1 entropy
            if len(dgms) > 1:
                H1_entropy.append(self.persistent_entropy(dgms[1]))
            else:
                H1_entropy.append(0.0)

            # H2 entropy
            if len(dgms) > 2:
                H2_entropy.append(self.persistent_entropy(dgms[2]))
            else:
                H2_entropy.append(0.0)

            # Plot PD + barcode every  steps
            if step == 1 or step % self.PrintPD_step == 0:
                fig, axs = plt.subplots(1, 3, figsize=(10, 4))
                fig.suptitle(f"Step {step}: Persistence Diagram & Barcodes ({representation_type})", fontsize=14)

                # Left: persistence diagrams for H0, H1, H2
                plot_diagrams(dgms, ax=axs[0], show=False)
                axs[0].set_title("Persistence Diagrams (H0, H1, H2)")

                # Center: barcodes
                self.barcode_plot(dgms, ax=axs[1])

                #Right: t_SNE projection at some training step
                axs[2].scatter(proj_2d[:, 0], proj_2d[:, 1], s=8)
                axs[2].set_title("t-SNE (2D)")
                axs[2].set_xlabel("t-SNE 1")
                axs[2].set_ylabel("t-SNE 2")

                plt.tight_layout()
                plt.show()

        #Plot persistent entropy for input vectors
        plt.figure(figsize=(8, 5))
        plt.plot(topo_steps, H0_entropy, label="H0 entropy")
        plt.plot(topo_steps, H1_entropy, label="H1 entropy")
        plt.plot(topo_steps, H2_entropy, label="H2 entropy")

        plt.xlabel("Training step")
        plt.ylabel("Persistent entropy")
        plt.title(f"Persistent Entropy vs Training Step ({representation_type})")
        plt.legend()
        plt.grid(True)
        plt.show()


# =================================
# Heatmap animation (layer weights)
# =================================
class HeatmapStuff:
    def __init__(self, config: dict):
        self.heatmap_step = config['heatmap_step'] 
        self.SaveTorF     = config['SaveTorF']
        self.latex_title  = config['latex_title']
        self.poly_name    = config['poly_name']
        self.p            = config['p']

    def HeatmapAnimation(self, representation_type: str, Matrix, Do_Fourier = False):
        # weight_mats: list of (out_dim, in_dim) tensors/arrays over training

        mats = []
        steps = []

        for i, W in enumerate(Matrix):
            step = i + 1
            if step != 1 and step % self.heatmap_step != 0:
                continue

            arr = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else np.array(W)
            arr = arr.reshape(arr.shape[0], -1)  # force 2D
            if Do_Fourier:
                arr = self._fft2_logmag(arr)
            mats.append(arr)
            steps.append(step)

        if len(mats) == 0:
            print("No frames collected (check heatmap_step / weight_mats).")
            return

        # global symmetric scale around 0 (good for weights)
        if Do_Fourier:
            vmin = 0.0
            vmax = max(float(np.max(A)) for A in mats)
        else:
            m = max(float(np.max(np.abs(A))) for A in mats)
            vmin, vmax = -m, m

        fig, ax = plt.subplots(figsize=(8, 4))
        cmap = "viridis" if Do_Fourier else "coolwarm"
        im = ax.imshow(mats[0], aspect="auto", vmin=vmin, vmax=vmax, cmap = cmap)
        plt.colorbar(im, ax=ax)

        ax.set_title(
            f"Heatmap {representation_type} {self.latex_title} mod {self.p} "
            f"(step {steps[0]}/{steps[-1]})"
        )
        if Do_Fourier:
            ny, nx = mats[0].shape

            kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
            ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny

            ax.set_xticks([0, nx//2, nx-1])
            ax.set_yticks([0, ny//2, ny-1])

            ax.set_xticklabels([int(kx[0]), 0, int(kx[-1])])
            ax.set_yticklabels([int(ky[0]), 0, int(ky[-1])])
        
            ax.set_xlabel("column frequency")
            ax.set_ylabel("row frequency")
            ax.invert_yaxis()
        else: 
            ax.set_xlabel("in index")
            ax.set_ylabel("out index")

        def update(k):
            im.set_data(mats[k])
            ax.set_title(
                f"Heatmap {representation_type} {self.latex_title} mod {self.p} "
                f"(step {steps[k]}/{steps[-1]})"
            )
            return im,

        anim = FuncAnimation(fig, update, frames=len(mats), interval=120, repeat=False)

        if self.SaveTorF:
            anim.save(
                f"heatmap_{representation_type}_{self.poly_name}_mod_{self.p}.mp4",
                writer="ffmpeg", dpi=150, fps=25
            )

        plt.close(fig)
        display(HTML(anim.to_jshtml()))

    def RowAnimation(self, representation_type: str, Matrix, row_idx: int, Columns = False):

        rows = []
        steps = []

        for i, W in enumerate(Matrix):
            step = i + 1
            if step != 1 and step % self.heatmap_step != 0:
                continue

            arr = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else np.array(W)
            arr = arr.reshape(arr.shape[0], -1)
            if Columns:       
                arr = arr.T   
            rows.append(arr[row_idx])
            steps.append(step)

        if len(rows) == 0:
            print("No frames collected.")
            return

        rows = np.array(rows)

        fig, ax = plt.subplots(figsize=(10,6))
        line, = ax.plot(rows[0])

        m = np.max(np.abs(rows))
        ax.set_ylim(-m, m)
        ax.set_xlabel("column index")
        ax.set_ylabel("weight value")

        if Columns:   
            Row_or_Column = 'column'
        else:
            Row_or_Column = 'row'

        ax.set_title(
            f"{representation_type} {Row_or_Column} {row_idx} {self.latex_title} mod {self.p} "
            f"(step {steps[0]}/{steps[-1]})"
        )

        def update(k):
            line.set_ydata(rows[k])
            ax.set_title(
                f"{representation_type} {Row_or_Column} {row_idx} {self.latex_title} mod {self.p} "
                f"(step {steps[k]}/{steps[-1]})"
            )
            return line,

        anim = FuncAnimation(fig, update, frames=len(rows), interval=120, repeat=False)

        if self.SaveTorF:
            anim.save(
                f"{Row_or_Column}_{row_idx}_{representation_type}_{self.poly_name}_mod_{self.p}.mp4",
                writer="ffmpeg", dpi=150, fps=25
            )

        plt.close(fig)
        display(HTML(anim.to_jshtml()))

    def _to_numpy_2d(self, X):
        """X: (p^2, H) tensor/array -> numpy (p^2, H)"""
        arr = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.array(X)
        arr = arr.reshape(arr.shape[0], -1)
        return arr

    def _neuron_grid_from_snapshot(self, snapshot_2d, neuron_idx: int):
        """
        snapshot_2d: (p^2, H) in canonical dataset order
        returns grid: (p, p) where grid[a,b] corresponds to input (a,b)
        """
        p = self.p
        if snapshot_2d.shape[0] != p * p:
            raise ValueError(f"Expected first dim p^2={p*p}, got {snapshot_2d.shape[0]}. "
                             f"Make sure you logged on dataset.dataset (full grid).")
        if not (0 <= neuron_idx < snapshot_2d.shape[1]):
            raise ValueError(f"neuron_idx out of range. Got {neuron_idx}, "
                             f"but hidden dim is {snapshot_2d.shape[1]}.")
        v = snapshot_2d[:, neuron_idx]          # (p^2,)
        grid = v.reshape(p, p)                  # canonical: index = a*p + b
        return grid

    def _fft2_logmag(self, grid, shift=True, detrend=True, eps=1e-12):
        """grid: (p,p) -> log(1+|FFT|) heatmap"""
        A = grid.astype(np.float64, copy=False)
        if detrend:
            A = A - A.mean()
        F = np.fft.fft2(A)
        if shift:
            F = np.fft.fftshift(F)
        return np.log1p(np.abs(F) + eps)

    def NeuronHeatmapAnimation(
        self,
        representation_type: str,      # "preactivations" or "activations"
        snapshots,                    # loss_dictionary['preactivations'] etc; list over training
        neuron_idx: int,
        Do_Fourier: bool = False,
        fft_shift: bool = True,
        fft_detrend: bool = True,
    ):
        """
        Makes an animation over training steps of ONE neuron's values over (a,b) grid.

        - snapshots[t] is (p^2, H) (because you logged on dataset.dataset) :contentReference[oaicite:3]{index=3}
        - grid[a,b] corresponds to input (a,b) because dataset is built in nested loops over a then b :contentReference[oaicite:4]{index=4}
        """
        mats = []
        steps = []

        for i, S in enumerate(snapshots):
            step = i + 1
            if step != 1 and step % self.heatmap_step != 0:
                continue

            arr = self._to_numpy_2d(S)                     # (p^2, H)
            grid = self._neuron_grid_from_snapshot(arr, neuron_idx)  # (p, p)

            if Do_Fourier:
                grid = self._fft2_logmag(grid, shift=fft_shift, detrend=fft_detrend)

            mats.append(grid)
            steps.append(step)

        if len(mats) == 0:
            print("No frames collected (check heatmap_step / snapshots).")
            return

        # color scaling
        if Do_Fourier:
            vmin = 0.0
            vmax = max(float(np.max(M)) for M in mats)
            cmap = "viridis"
        else:
            m = max(float(np.max(np.abs(M))) for M in mats)
            vmin, vmax = -m, m
            cmap = "coolwarm"

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mats[0], aspect="equal", vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, ax=ax)

        base = f"{representation_type} neuron {neuron_idx}"
        if Do_Fourier:
            base = f"FFT[{representation_type}] neuron {neuron_idx}"

        if Do_Fourier:
            ny, nx = mats[0].shape

            kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
            ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny

            ax.set_xticks([0, nx//2, nx-1])
            ax.set_yticks([0, ny//2, ny-1])

            ax.set_xticklabels([int(kx[0]), 0, int(kx[-1])])
            ax.set_yticklabels([int(ky[0]), 0, int(ky[-1])])
        
            ax.set_xlabel("column frequency")
            ax.set_ylabel("row frequency")
            ax.invert_yaxis()
        else: 
            ax.set_xlabel("b")
            ax.set_ylabel("a")


        def update(k):
            im.set_data(mats[k])
            ax.set_title(
                f"Heat map {base} for {self.latex_title} mod {self.p} "
                f"(step {steps[k]}/{steps[-1]})"
            )
            return im,

        anim = FuncAnimation(fig, update, frames=len(mats), interval=120, repeat=False)

        if self.SaveTorF:
            tag = "fft_" if Do_Fourier else ""
            anim.save(
                f"{tag}{representation_type}_neuron_{neuron_idx}_{self.poly_name}_mod_{self.p}.mp4",
                writer="ffmpeg", dpi=150, fps=25
            )

        plt.close(fig)
        display(HTML(anim.to_jshtml()))
