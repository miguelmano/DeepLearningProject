#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
from matplotlib import pyplot as plt


import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        scores = self.W.dot(x_i)
        predicted_label = scores.argmax()
        if predicted_label != y_i:
            self.W[y_i] += x_i
            self.W[predicted_label] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        scores = np.exp(self.W.dot(x_i))
        scores /= np.sum(scores)
        scores[y_i] -= 1
        self.W -= learning_rate * np.outer(scores, x_i)

class MLP(object):
    # Q1.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers=1):
        # Initialize an MLP with a single hidden layer.
        mean = 0.1
        std = 0.1
        if layers == 0:
            self.W = [np.random.normal(mean, std, (n_classes, n_features))]
        else:
            self.W = [np.random.normal(mean, std, (hidden_size, n_features))]
            self.W += [np.random.normal(mean, std, (hidden_size, hidden_size)) for i in range(layers - 1)]
            self.W += [np.random.normal(mean, std, (n_classes, hidden_size))]

        self.b = [np.zeros(hidden_size) for i in range(layers)] + [np.zeros(n_classes)]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        H = X.T
        for W_i, b_i in zip(self.W[:-1], self.b[:-1]):
            Z = W_i.dot(H) + b_i.reshape(-1, 1)
            H = Z.clip(0)
        scores = self.W[-1].dot(H) + self.b[-1].reshape(-1, 1)
        return scores.argmax(axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        epoch_loss = 0.0
        for x_i, y_i in zip(X, y):
            z = []
            h = []
            y_hat = x_i
            for W_i, b_i in zip(self.W[:-1], self.b[:-1]):
                z_i = W_i.dot(y_hat) + b_i
                y_hat = z_i.clip(0)
                z.append(z_i)
                h.append(y_hat)
            y_hat = self.W[-1].dot(y_hat) + self.b[-1]

            dz = np.exp(y_hat - y_hat.max())
            dz /= np.sum(dz)
            dz[y_i] -= 1

            for W_i, b_i, z_i, h_i in zip(reversed(self.W), reversed(self.b), reversed([np.ones(x_i.shape[0])] + z), reversed([x_i] + h)):
                dW = np.outer(dz, h_i)
                db = dz
                dh = W_i.T.dot(dz)
                dz = dh * (z_i > 0)
                W_i -= learning_rate * dW
                b_i -= learning_rate * db
                
            if (dz[y_i] > 0):
                epoch_loss += -np.log(dz[y_i])
        
        return epoch_loss / X.shape[0]
        
def plot(epochs, train_accs, val_accs, name):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

def plot_loss(epochs, loss, name):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.savefig('%s.pdf' % name, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
        name = 'hw1-q1-1a perceptron'
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
        name = 'hw1-q1-1b logistic regression ' + str(opt.learning_rate)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
        name = 'hw1-q1-2b mlp'
        
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]

        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, name)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, 'hw1-q1-2b mlp loss')


if __name__ == '__main__':
    main()
