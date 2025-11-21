# To add noise to the inputs, we can try the following:
# choose whether to add noise to both train and test data
# given a dataset of the form (a, b, c)
# inputs are set of (a,b), then for each (a,b) replace
# a -> a + epsilon, b -> b + epsilon (epsilon is different for a, b) and
# each instance
# note, this should be done in the one-hot embedder, so
# a is a p-vector, epsilon is a p-vector in N(0, sigma * eye(p)), sigma is a
# hyperparameter
# note that the a in (a, b) and (a, b') get mapped to different p-vectors
# since we add noise for each instance
# if we added same noise to each instance, then it'd be like a fixed embedding
# c remains integer valued (could imagine adding noise to it)

# for negative sampling, we can follow the original work closely
# a slightly more general approach is the following
# check that it includes original approach
# data is in the form (a, b, c): p ** 3 possible data points; however,
# only p ** 2 are true while other p ** 2 (p - 1) are false; model
# should predict if some data is true or false
# we think of (a,b) as the "context" and c as the "word," appearing in given
# "context"
# original approach is something like:
# get a context vector v_(a,b) = f(a,b); f could be nonlinear map or
# an average v = (a+b)/2)
# look at overlap = v @ c (note we implicitly require len(v) = len(c)
# model probability with sigmoid(overlap)
# also consider some negative samples for which model should produce false,
# for a given context; model should max sigmoid(- overlap_neg)
# take sum of negative log likelihoods and minimize over parameters

# another approach
# take all (a, b, c): split into true and false examples
# create some MLP (or other architecture, e.g., like sentiment analysis)
# output of MLP can be 2 classes, either true or false

# how does prediction work
# imagine given (a,b) and we want to get c
# there are p possible values of c
# calculate (a, b, c) for all p values of c and say model predicts c for
# which the true value is highest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional  


def print_losses(max_steps: int,
                 step: int,
                 loss_dictionary: dict,
                 num_print_steps: int = 4):
    if step == 1 or step % (max_steps // num_print_steps) == 0:
        print(f"step {step}/{max_steps}:\t " +
              f"train_loss = {loss_dictionary['train_loss'][-1]: .2f},\t")


class ActPoly:
    def __init__(self, m: float = 2.):
        self.m = m

    def __call__(self, x):
        return x ** self.m


class NegMLP(nn.Module):
    def __init__(self,
                 p,
                 embedding_dim,
                 mlp_dim,
                 activation=ActPoly(m=2),
                 model_seed=8675309):
        super().__init__()
        self.p = p
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.activation = activation
        torch.manual_seed(model_seed)
        self.embed = nn.Embedding(p, embedding_dim)
        self.linear1 = nn.Linear(2 * embedding_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, embedding_dim)
        with torch.no_grad():
            self.embed.weight.uniform_(-1.0 / np.sqrt(self.p), 1.0 / np.sqrt(self.p))
        self.loss_dictionary = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 
                                'test_accuracy': [], 'positive_train_accuracy': [], 
                                'negative_train_accuracy': [],'positive_test_accuracy': [], 
                                'negative_test_accuracy': [], 'counts_hist_all': torch.zeros(self.p, dtype=int), 
                                'prediction_mike': torch.zeros(self.p, dtype=int), 'sorta_prob_dist': []}

    def forward(self, x):
        # Shape of x: (b, 3)
        b, _ = x.shape
        x = self.embed(x) # Shape of x: (b, 3, embedding_dim)
        y = x[:, -1, :]
        x = x[:, :2, :]
        x = x.reshape(b, -1) # Concatenates x to have shape: (b, 2 * embedding_dim)
        x = self.linear1(x) # Shape of x: (b, mlp_dim)
        x = self.activation(x)
        x = self.linear2(x) # Shape of x: (b, embedding_dim)
        # The following two lines implement batch dot product using @ operator
        x = x[:, None, :] # Shape of x: (b, 1, embedding_dim)
        y = y[..., None] # Shape of y: (b, embedding_dim, 1)
        return (x @ y).squeeze() # Shape of out: (b)


class NegTrainer:
    """Training class."""
    def __init__(self,
                 optimizer_name="Adam",
                 learning_rate=0.01,
                 num_negs_per_example = -1):
        self.optimizer_name = optimizer_name
        self.optimizer = None
        self.learning_rate = learning_rate
        self.num_negs_per_example = num_negs_per_example

    def optimizer_chooser(self, model, weight_decay=0.0):
        if self.optimizer_name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        if self.optimizer_name == 'AdamW':  # optional but recommended
            return torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        raise ValueError("Try again: optimizer options are 'Adam', 'AdamW', and 'SGD'.")
    
    #This function evaluates the model over the positive and negative examples in the train data only! Updates the train accuracy
    def evaluate_train(self, model, train_examples, labels_train, a , b):
        model.eval()
        with torch.no_grad():
            outputs = model(train_examples)
            probs = torch.sigmoid(outputs) 
            preds = (probs >= 0.5).int()
            accuracy = (preds == labels_train).float().mean().item()
            positive_accuracy = (preds[:a] == labels_train[:a]).float().mean().item()
            negative_accuracy = (preds[-b:] == labels_train[-b:]).float().mean().item()
            model.loss_dictionary['train_accuracy'].append(accuracy)
            #Accuracy only on positives:
            model.loss_dictionary['positive_train_accuracy'].append(positive_accuracy)
            #Accuracy only on negatives:
            model.loss_dictionary['negative_train_accuracy'].append(negative_accuracy)

    #This function evaluates the model over the positive and negative examples in the train data only! Updates the train accuracy
    def evaluate_test(self, model, test_examples, labels_test, c, d):
        model.eval()
        with torch.no_grad():
            outputs = model(test_examples)
            probs = torch.sigmoid(outputs)  
            preds = (probs >= 0.5).int()
            accuracy = (preds == labels_test).float().mean().item()
            positive_accuracy = (preds[:c] == labels_test[:c]).float().mean().item()
            negative_accuracy = (preds[-d:] == labels_test[-d:]).float().mean().item()
            model.loss_dictionary['test_accuracy'].append(accuracy)
            #For test loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels_test)
            model.loss_dictionary['test_loss'].append(loss.item())
            #Accuracy only on positives:
            model.loss_dictionary['positive_test_accuracy'].append(positive_accuracy)
            #Accuracy only on negatives:
            model.loss_dictionary['negative_test_accuracy'].append(negative_accuracy)

            #In the case of our experiment where we restrict the test data to a single data point, we ask our model to make a prediction for which (a,b,c) is correct.
            #Multiple (a,b,c) triples may be predicted as correct. So we have a histogram where we log all triples counted true, and a histogram where we collect the triple with greatest probability
            if c == 1:
                last_elements = test_examples[:, -1]
                sort_indices = torch.argsort(last_elements)
                test_examples = test_examples[sort_indices]
                preds = preds[sort_indices]
                probs = probs[sort_indices]
                model.loss_dictionary['counts_hist_all'] += preds.detach().cpu()
                max_prob = torch.max(probs)
                model.loss_dictionary['prediction_mike'] += torch.where(probs == max_prob, 1, 0).detach().cpu()
                #Produce sorta probability distribution
                sum = torch.sum(probs)
                prob_dist = probs / sum
                model.loss_dictionary['sorta_prob_dist'].append(prob_dist.detach().cpu())
            




    def train_model(self,
                    model,
                    dataset,
                    max_steps=1,
                    batch_size=-1,
                    weight_decay = 0, 
                    print_loss=True,
                    train_seed=2016,
                    device = None):
        
        #Run on gpu
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        """Train model and return loss dictionary."""
        self.optimizer = self.optimizer_chooser(model, weight_decay=weight_decay)
        pos_examples = dataset.train_data
        neg_examples_dict = dataset.neg_examples
        batch_size = batch_size if batch_size != -1 and batch_size < dataset.num_train else dataset.num_train

        #Get the negatives for evaluation on train
        negs_for_train = []
        for pos_example in pos_examples.tolist():
            all_negs = neg_examples_dict[tuple(pos_example)]
            neg_ids = torch.randperm(len(all_negs))
            negatives = all_negs[neg_ids]
            negs_for_train.append(negatives)
        
        # Stack negatives into one tensor for train
        negs_for_train = torch.cat(negs_for_train, dim=0)   # shape: (total_negs, d)
        train_examples = torch.cat([pos_examples, negs_for_train], dim=0).to(device)  # shape: (num_pos + total_negs, d)
        labels_train = torch.cat([
            torch.ones(len(pos_examples), dtype=torch.float),
            torch.zeros(len(negs_for_train), dtype=torch.float)
        ]).to(device)
        a = len(pos_examples)
        b = len(negs_for_train)

        #Get the negatives for evaluation on test
        pos_for_test = dataset.test_data
        negs_for_test = []
        for pos_example in pos_for_test.tolist():
            all_negs = neg_examples_dict[tuple(pos_example)]
            neg_ids = torch.randperm(len(all_negs))
            negatives = all_negs[neg_ids]
            negs_for_test.append(negatives)


        # Stack negatives into one tensor for test
        negs_for_test = torch.cat(negs_for_test, dim=0)   # shape: (total_negs, d)
        test_examples = torch.cat([pos_for_test, negs_for_test], dim=0).to(device)  # shape: (num_pos + total_negs, d)
        labels_test = torch.cat([
            torch.ones(len(pos_for_test), dtype=torch.float),
            torch.zeros(len(negs_for_test), dtype=torch.float)
        ]).to(device)
        c = len(pos_for_test)
        d = len(negs_for_test)
        #counts_hist_all = torch.zeros(c+d)
        #prediction_mike = torch.zeros(c+d)

        # Training loop
        torch.manual_seed(train_seed)
        for i in range(max_steps):
            # Bootstrap training using empirical distribution
            idx = torch.randperm(dataset.num_train)
            pos_examples_batch = pos_examples[idx][:batch_size].to(device)
            neg_examples_batch = self.get_negatives(pos_examples_batch, neg_examples_dict).to(device)
            self.fit(model, pos_examples_batch, neg_examples_batch)
            self.evaluate_train(model, train_examples, labels_train, a , b)
            self.evaluate_test(model, test_examples, labels_test, c , d)

            if print_loss:
                print_losses(max_steps, i + 1, model.loss_dictionary)

    def get_negatives(self, pos_set, neg_examples_dict):
        max_negs_per_example = list(neg_examples_dict.values())[0].shape[0]
        num_negs_per_example = max_negs_per_example if self.num_negs_per_example == -1 else self.num_negs_per_example
        neg_examples = []
        for pos_example in pos_set.tolist():
            all_negs = neg_examples_dict[tuple(pos_example)]
            neg_ids = torch.randperm(len(all_negs))
            negatives = all_negs[neg_ids][:num_negs_per_example]
            neg_examples.append(negatives)
        return torch.cat(neg_examples, dim=0)

    def fit(self, model, pos_set, neg_set):
        """Perform one optimization step."""
        model.train()
        p, n = pos_set.shape[0], neg_set.shape[0]
        pos_loss = - torch.log(torch.sigmoid(model(pos_set))).sum()
        neg_loss = - torch.log(torch.sigmoid(- model(neg_set))).sum()
        # pos = torch.sigmoid(model(pos_set))
        # neg = torch.sigmoid(model(neg_set))
        total_loss = (pos_loss + (p / n) * neg_loss) / (2 * p) # Weighted: pos and neg contribute equally
        # total_loss = (pos_loss + neg_loss) / (n + p) # Unweighted
        # total_loss = (((1 - pos) ** 2).sum() + (neg ** 2).sum())/(n + p) # like an L2 loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        model.loss_dictionary['train_loss'].append(total_loss.detach().clone().item())

def accuracy(pos_output=None, neg_output=None):
    with torch.no_grad():
        pos_and_neg = []
        if pos_output is not None:
            pos = (torch.sigmoid(pos_output.clone()) >= .50).float()
            pos_and_neg.append(pos)
        if neg_output is not None:
            neg = (torch.sigmoid(neg_output.clone()) < .50).float()
            pos_and_neg.append(neg)
        results = torch.cat(pos_and_neg, dim=0)
        return results.mean()


class NegOverlap(nn.Module):
    def __init__(self,
                 p,
                 embedding_dim,
                 learned_average=False,
                 model_seed=8675309):
        super().__init__()
        self.p = p
        self.embedding_dim = embedding_dim
        self.learned_average = learned_average
        torch.manual_seed(model_seed)
        self.embed = nn.Embedding(p, embedding_dim)
        self.average = nn.Linear(2 * embedding_dim, embedding_dim)
        with torch.no_grad():
            self.embed.weight.uniform_(-1.0 / np.sqrt(self.p), 1.0 / np.sqrt(self.p))
        self.loss_dictionary = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}

    def forward(self, x):
        # Shape of x: (b, 3)
        b, _ = x.shape
        x = self.embed(x) # Shape of x: (b, 3, embedding_dim)
        y = x[:, -1, :]
        x = x[:, :2, :]
        if self.learned_average:
            x = x.reshape(b, -1)  # Concatenates x to have shape: (b, 2 * embedding_dim)
            x = self.average(x)  # Shape of x: (b, embedding_dim)
        else:
            x = (x[:, 0, :] + x[:, 1, :]) / 2
        # The following two lines implement batch dot product using @ operator
        x = x[:, None, :] # Shape of x: (b, 1, embedding_dim)
        y = y[..., None] # Shape of y: (b, embedding_dim, 1)
        return (x @ y).squeeze() # Shape of out: (b)