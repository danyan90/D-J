import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from torch.nn import MSELoss, CrossEntropyLoss
import matplotlib.pyplot as plt


def accuracy_function(y_hat, one_hot_y):
    """Compute the accuracy of prediction y_hat vs ground truth one_hot_y."""
    b, p = y_hat.shape
    y_hat = torch.argmax(y_hat, dim=1, keepdim=False) # Shape of y_hat = (B)
    one_hot_y_hat = functional.one_hot(y_hat, num_classes=p) * 1.0 # Shape of one_hot_y_hat = (B, p)
    return (1 / b) * (one_hot_y_hat * one_hot_y).sum()


def optimizer_chooser(model, name, rate, weight_decay=0.0):
    if name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=rate, weight_decay=weight_decay)
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=rate, weight_decay=weight_decay)
    if name == 'AdamW':  # optional but recommended
        return torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=weight_decay)
    raise ValueError("Try again: optimizer options are 'Adam', 'AdamW', and 'SGD'.")


def scheduler_chooser(optimizer, anneal_time):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, anneal_time)


def criterion_chooser(criterion_name):
    if criterion_name == 'MSE':
        return MSELoss(reduction='mean')
    if criterion_name == 'CrossEntropy':
        return CrossEntropyLoss()
    raise ValueError("Try again: criterion_name options are 'MSE' and 'CrossEntropy'.")


def print_losses_and_accuracies(max_steps: int,
                                step: int,
                                loss_dictionary: dict,
                                num_print_steps: int = 4):
    if step % (max_steps // num_print_steps) == 0:
        print(f"step {step}/{max_steps}:\t " +
              f"dirty_loss = {loss_dictionary['train_loss'][-1]: .2f},\t " +
              f"dirty_acc = {loss_dictionary['train_accuracy'][-1]: .2f},")
        print(f"step {step}/{max_steps}:\t " +
              f"test_loss = {loss_dictionary['test_loss'][-1]: .2f},\t " +
              f"test_acc = {loss_dictionary['test_accuracy'][-1]: .2f}")
        print(f"step {step}/{max_steps}:\t " +
              f"clean_loss = {loss_dictionary['other_loss'][-1]: .2f},\t " +
              f"clean_acc = {loss_dictionary['other_accuracy'][-1]: .2f}")


def make_time_dict(step, loss_dictionary, time_dict, start_acc=.10, end_acc=.99):
    """Determine memorization/generalization start and end times and record in time_dict."""
    if loss_dictionary['train_accuracy'][-1] > start_acc and time_dict['mem'][0] == float('inf'):
        time_dict['mem'][0] = float(step)
    if loss_dictionary['train_accuracy'][-1] > end_acc and time_dict['mem'][1] == float('inf'):
        time_dict['mem'][1] = float(step)
    if loss_dictionary['test_accuracy'][-1] > start_acc and time_dict['gen'][0] == float('inf'):
        time_dict['gen'][0] = float(step)
    if loss_dictionary['test_accuracy'][-1] > end_acc and time_dict['gen'][1] == float('inf'):
        time_dict['gen'][1] = float(step)


def plot_losses_and_accuracies(loss_dictionary: dict,
                               plot_losses=False,
                               output_name=None):
    plt.plot(loss_dictionary['train_accuracy'], label='train accuracy', color='tab:red')
    plt.plot(loss_dictionary['test_accuracy'], label='test accuracy', color='tab:blue')
    if len(loss_dictionary['other_accuracy']) > 0:
        plt.plot(loss_dictionary['other_accuracy'], label='clean train accuracy', color='tab:green')
    if plot_losses:
        plt.plot(loss_dictionary['train_loss'], label='train loss', color='tab:red', linestyle='dashed')
        plt.plot(loss_dictionary['test_loss'], label='test loss', color='tab:blue', linestyle='dashed')
        if len(loss_dictionary['other_loss']) > 0:
            plt.plot(loss_dictionary['other_loss'], label='clean train loss', color='tab:green', linestyle='dashed')
        plt.title('Losses and Accuracies vs Optimization Steps')
        plt.ylabel('Loss/Accuracy')
    else:
        plt.title('Accuracy vs Optimization Step')
        plt.ylabel('Accuracy')
    plt.xlabel('Optimization Step')
    plt.legend()
    if output_name:
        plt.savefig(output_name)
    plt.show()


class AppendixCFunction:
    #Here we define a modular polynomial of the form (c1*x^d1 + c2*x^d2)^d3 + c3*x^d4*x^d5. So we take c = [c1 c2 c3] and d = [d1 d2 d3 d4 d5]
    def __init__(self, a: int, b: int, p: int):
        self.a = a
        self.b = b
        self.p = p

    def __call__(self, x: int, y: int):
        a1, a2, a3 = self.a
        b1, b2, b3, b4, b5 = self.b
        return ((a1*(x**b1) + a2*(y**b2))**b3 + a3*(x**b4)*(y**b5)) % self.p



class LinearFunction:
    """"Linear two-variable modular polynomial."""
    def __init__(self, a: int, b: int, p: int):
        self.a = a
        self.b = b
        self.p = p

    def __call__(self, x: int, y: int) -> int:
        return (self.a * x + self.b * y) % self.p


class ThreeTermNonlinearFunction:
    """Nonlinear two-variable modular polynomial with three terms."""
    def __init__(self, a: list[int], m: list[int], p: int):
        assert len(a) == 3, "Should contain 3 integers."
        assert len(m) == 4, "Should contain 4 integers."
        self.a = a
        self.m = m
        self.p = p

    def __call__(self, x: int, y: int) -> int:
        a1, a2, a3 = self.a
        m1, m2, m3, m4 = self.m
        first_term = a1 * (x ** m1)
        second_term = a2 * (x ** m2) * (y ** m3)
        third_term = a3 *  (y ** m4)
        return (first_term + second_term + third_term) % self.p


class RandomFunction:
    """Choose a random number over p values, given two-variable input."""
    def __init__(self, p: int, rng_seed=1982):
        self.p = p
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)

    def __call__(self, x: int, y: int) -> int:
        return self.rng.choice(self.p, size=1).item()


class DataObject:
    """Dataset class for a given (linear) modular polynomial."""
    def __init__(self,
                 modular_function,
                 split=.50,
                 randomize=True,
                 corruption_level = 0.0,
                 corruption_function = None,
                 data_seed=8675309,
                 corruption_seed=217,
                 noise_seed=5882300):
        self.function = modular_function
        self.p = modular_function.p
        self.split = split
        self.randomize = randomize
        self.corruption_level = corruption_level
        self.corruption_function = corruption_function
        self.data_seed = data_seed
        self.corruption_seed = corruption_seed
        self.noise_seed = noise_seed
        self.random_function = RandomFunction(p=self.p, rng_seed=self.corruption_seed)
        # self.nearby_function = LinearFunction(self.a, self.b + 1, self.p)
        # self.function = LinearFunction(self.a, self.b, self.p)
        self.dataset = self.make_dataset()
        self.data_length = len(self.dataset)
        self.train_data, self.test_data = self.train_test_split()
        self.num_train = len(self.train_data)
        if corruption_function:
            self.corrupt_data = self.corrupt(corruption_level=corruption_level, corruption_function=corruption_function)
        else:
            self.corrupt_data = self.corrupt(corruption_level=corruption_level, corruption_function=self.random_function)
        self.neg_examples = self.make_neg_examples()

    def make_dataset(self):
        """Make the dataset."""
        examples = []
        for x in range(self.p):
            for y in range(self.p):
                z = self.function(x, y)
                examples.append((x, y, z))
        return torch.tensor(examples)

    def train_test_split(self):
        """Split the dataset into train and test sets."""
        train_size = int(self.split * self.data_length)
        if not self.randomize:
            train_data, test_data = self.dataset[:train_size], self.dataset[train_size:]
            return train_data, test_data
        torch.manual_seed(self.data_seed)
        idx = torch.randperm(self.data_length)
        perm_data = self.dataset[idx]
        train_data, test_data = perm_data[:train_size], perm_data[train_size:]
        return train_data, test_data

    def corrupt(self, corruption_level=0.0, corruption_function=None):
        """Corrupt {corruption_level} of training data using {corrupting_function}."""
        num_corrupted = int(corruption_level * self.num_train)
        corrupted_train = []
        for i, example in enumerate(self.train_data):
            if i < num_corrupted:
                a, b, _ = example
                c = torch.tensor([corruption_function(a.item(), b.item())])
                corrupted_example = torch.tensor([a, b, c])
                corrupted_train.append(corrupted_example)
            else:
                corrupted_train.append(example)
        return torch.stack(corrupted_train, dim=0)

    def make_neg_examples(self):
        """Create dictionary of negative examples."""
        neg_examples = {}
        for example in self.dataset:
            a, b, c = example.tolist()
            neg_examples[(a, b, c)] = torch.tensor([[a, b, c_prime] for c_prime in range(self.p) if c_prime != c])
        return neg_examples

    def add_noise(self, only_train=True):
        pass


class Trainer:
    """Training class."""
    def __init__(self,
                 optimizer_name="Adam",
                 criterion_name="MSE",
                 use_cos_scheduler=False,
                 num_maxs=0,
                 learning_rate=0.001):
        self.optimizer_name = optimizer_name
        self.use_cos_scheduler = use_cos_scheduler
        self.learning_rate = learning_rate
        self.criterion_name = criterion_name
        self.criterion = criterion_chooser(criterion_name)
        self.num_maxs = num_maxs

    def train_model(self,
                    model,
                    dataset,
                    max_steps=1,
                    batch_size=-1,
                    weight_decay = 0,
                    print_losses=False,
                    plot_it=False,
                    quick_train=False,
                    train_seed=2016,
                    device = None):
        """
        Train model and return loss/accuracy and time dictionaries.
        {batch_size} = -1 or > {len(dataset.train_data)} uses full batch for training.
        {quick_train} ends training when test accuracy is greater than .99.
        """
        #Run on gpu
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.criterion.to(device)

        optimizer = optimizer_chooser(model, self.optimizer_name, self.learning_rate, weight_decay=weight_decay)
        if self.use_cos_scheduler:
            # {anneal_time_multiplier} hardcodes of how fast to decrease learning rate over {max_steps}
            anneal_time_multiplier = 2
            scheduler = scheduler_chooser(optimizer, anneal_time_multiplier * max_steps)
        else:
            scheduler = None
        if dataset.corruption_level > 0:
            train_data = dataset.corrupt_data.to(device)
            other_data = dataset.train_data.to(device)
        else:
            train_data = dataset.train_data.to(device)
            other_data = None
        batch_size = batch_size if batch_size != -1 and batch_size < dataset.num_train else dataset.num_train

        torch.manual_seed(train_seed)
        for i in range(max_steps):
            # bootstrap training using empirical distribution
            idx = torch.randperm(dataset.num_train)
            train_data_batch = train_data[idx][:batch_size]
            self.fit(model, train_data_batch, optimizer, scheduler)
            self.evaluate(model, dataset.test_data.to(device), other_set=other_data)
            make_time_dict(i+1, model.loss_dictionary, model.time_dictionary)

            if print_losses:
                print_losses_and_accuracies(max_steps, i+1, model.loss_dictionary)
            if quick_train:
                if model.loss_dictionary['test_accuracy'][-1] > .99:
                    break
        if plot_it:
            plot_losses_and_accuracies(model.loss_dictionary)

    def fit(self, model, train_set, optimizer, scheduler):
        """Perform one optimization step."""
        model.train()
        loss, _ = self.get_loss_and_accuracy(model, train_set)
        if self.num_maxs > 0:
            first_minors = model.get_first_minors(train_set[:, :2], self.num_maxs)
            minors_mse = 10e6 * sum(minor ** 2 for minor in first_minors) / len(first_minors)
        else:
            minors_mse = 0
        # print(loss, minors_mse)
        #train_loss = loss + minors_mse + 0 * (model.f1.weight ** 2).mean()
        train_loss = loss + minors_mse
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        _, train_accuracy = self.get_loss_and_accuracy(model, train_set)

        model.loss_dictionary['train_loss'].append(train_loss.detach().clone().item())
        model.loss_dictionary['train_accuracy'].append(train_accuracy.detach().clone().item())

    def evaluate(self, model, test_set, other_set=None):
        """
        Evaluate model on test set and other set.
        {other_set} can be the un-corrupted training set.
        """
        model.eval()
        with torch.no_grad():
            test_loss, test_accuracy = self.get_loss_and_accuracy(model, test_set)
            model.loss_dictionary['test_loss'].append(test_loss.detach().clone().item())
            model.loss_dictionary['test_accuracy'].append(test_accuracy.detach().clone().item())
            if other_set is not None:
                other_loss, other_accuracy = self.get_loss_and_accuracy(model, other_set)
                model.loss_dictionary['other_loss'].append(other_loss.detach().clone().item())
                model.loss_dictionary['other_accuracy'].append(other_accuracy.detach().clone().item())
            if len(test_set) == 1:
                #Overlap.eval()
                logits = model(test_set[:, :2])
                probs = torch.softmax(logits, dim = 1)  # convert to probability distribution #dim = 1
                pred_class = torch.argmax(probs, dim=1).item()
                #max_prob = torch.max(probs)
                model.loss_dictionary['prediction'].append(pred_class)
                #model.loss_dictionary['counts_hist'] += torch.where(probs == max_prob, 1, 0)  #.cpu().detach().numpy()
                model.loss_dictionary['counts_hist'][pred_class] += 1
                model.loss_dictionary['prob_dist'].append(probs)

    def get_loss_and_accuracy(self, model, data):
        """Compute losses and accuracies in fit and evaluate functions."""
        # Shape of data: (B, 3)
        x, y = data[:, :2], data[:, -1].long()
        y_hat = model(x)
        num_classes = y_hat.detach().clone().shape[1]
        one_hot_y = functional.one_hot(y, num_classes=num_classes) * 1.0
        if self.criterion_name == 'CrossEntropy':
            loss = self.criterion(y_hat, y)
        else:
            # Multiplication by model.p=p gives mean over batches, not all tensor elements
            loss = self.criterion(y_hat, one_hot_y) * model.p
        accuracy = accuracy_function(y_hat.detach().clone(), one_hot_y)
        #print(loss)
        #print(accuracy) 
        return loss, accuracy


    def get_exact_hessian(self, model, data, make_plot=True):
        """Compute Hessian and its eigenvalues."""
        loss = self.get_loss_and_accuracy(model, data)[0]
        param_list = [param for param in model.parameters()]
        for param in param_list:
            param.grad = None
        d1 = torch.autograd.grad(loss, param_list, create_graph=True, retain_graph=True)
        grad_collection = [grad.view(-1) for grad in d1]
        flat_grads = torch.cat(grad_collection, dim=0)
        second_grads = []
        for g in flat_grads:
            d2 = torch.autograd.grad(g, param_list, retain_graph=True)
            second_grads.append(d2)
        hess = []
        for tup in second_grads:
            flat_tup = []
            for param_g in tup:
                param_g = param_g.view(-1)
                flat_tup.append(param_g)
            flat_tup = torch.cat(flat_tup, dim=0)
            hess.append(flat_tup)
        hess = torch.stack(hess, dim=0)
        eigs_approx = torch.linalg.eigvals(hess)
        eigs = torch.real(eigs_approx) # Remove imaginary component due to numerical imprecision
        eigs, _ = torch.sort(eigs, descending=False)
        if make_plot:
            plt.plot(eigs, marker='o', color='tab:blue')
            plt.xlabel('order')
            plt.ylabel('eigenvalue')
            plt.title('Hessian Eigenvalues in Ascending Order')
            plt.show()

        return hess, eigs


class Activation(nn.Module):
    def forward(self, x):
        return x ** 2
    
class Overlap(nn.Module):
    def __init__(self, p, embedding_dim, hidden, model_seed=8675309):
        super().__init__()
        self.p = p
        torch.manual_seed(model_seed)
        self.embed = nn.Embedding(p, embedding_dim)

        self.proj = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden),
            Activation(),
            nn.Linear(hidden, p)
        )

        self.loss_dictionary = {
            'train_loss': [], 'train_accuracy': [],
            'test_loss': [],  'test_accuracy': [],
            'other_loss': [], 'other_accuracy': [],
            'counts_hist': torch.zeros(self.p, dtype=int), 'prob_dist': [],
            'prediction': [],
        }
        self.time_dictionary = {'mem': [float('inf'), float('inf')],
                                'gen': [float('inf'), float('inf')]}

    def forward(self, x):
        # x: (B, 2)
        e = self.embed(x)         # (B, 2, d)
        e1, e2 = e[:, 0, :], e[:, 1, :]
        #h = torch.cat([e1, e2, e1 * e2], dim=-1)  # <-- interaction term
        h = torch.cat([e1, e2], dim=-1)  # <-- no interaction term
        logits = self.proj(h)     # (B, p)
        return logits
    
