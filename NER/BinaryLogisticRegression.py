from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418 at KTH.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.1  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 1000 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y
        
        :param      x:    The input features
        :param      y:    The correct labels
        """

        # x is only one row from the self.x
        yhat = self.sigmoid(np.dot(x, self.theta))

        # use the loss formula L = -(y*ln(yhat) + (1-y)*ln(1-yhat))
        loss = np.dot(y, np.log(yhat)) + np.dot((1-y), np.log(1-yhat))

        # Take the average and negative
        return loss * (-1/self.DATAPOINTS)


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # Dot product of thetaT (the weights) and x[i] (features)
        dot_prod = np.dot(self.x[datapoint], self.theta)

        sigmoid = self.sigmoid(dot_prod)

        if label == 1:
            return sigmoid
        elif label == 0:
            return 1 - sigmoid


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # Hämta alla gissningar alltså typ conditional_prob, skapar en lista, m lång, tar varje feature gånger varje theta och adderar dem, sedan sigmoid för ett värde mellan 0-1
        guesses = self.sigmoid(np.dot(self.x, self.theta))

        # Summera alla self.x * guess(sigma) - lösningen
        self.gradient = np.dot(self.x.T, guesses - self.y) / self.DATAPOINTS




    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """

        # exakt samma som compute all men som tur använder vi numpy som så vi kan bara lägga in en lista som parameter till en 2d matris och få ut exakt de radernas värden,
        guesses = self.sigmoid(np.dot(self.x[minibatch], self.theta))

        # Summera alla self.x[minibatch] * (guess(sigma) - lösningen) sedan dela med antalet element vi har kollat för istället för alla som i compute all
        self.gradient = np.dot(self.x[minibatch].T, guesses - self.y[minibatch]) / len(minibatch)


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # Check if random word is a name or not using y

        # multiply each weight with its maching feauture and add them together
        # x_times_theta is also more commonly simply called z, its what we put in sigmoid function to get a probability between 0 and 1.
        x_times_theta = np.dot(self.x[datapoint], self.theta)

        #peform guess, this value is somewhere between 0 and 1, closer to 1 if model thinks there is a high probabilty for this word to be a name
        guess = self.sigmoid(x_times_theta)
        # Check if random word is a name or not using y
        error = guess - self.y[datapoint]
        # each feature gets multilied by error, if feature was not there for example not capitalized error gets multiplied and only affecting features values are nudged
        self.gradient = self.x[datapoint] * error



    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        theta_change_total = float('inf')
        # we cannot use previous while loop logic since if we randomly stump on easy word thats model gets almost 99% right we will got super low gradient and training will stop early
        for i in range(self.MAX_ITERATIONS):
            random_word_index = np.random.randint(0, self.DATAPOINTS)
            self.compute_gradient(random_word_index)
            self.theta = self.theta - self.LEARNING_RATE*self.gradient

            # Euclidian distance of how much theta changes, close to 0 means we have converged, squared sum of all
            theta_change_total = np.sum(self.gradient ** 2)

            # Plot the loss 50 times iterations, we increased the max iterations cause it gave way to bad results
            if i % (self.MAX_ITERATIONS / 50) == 0:
                self.update_plot(self.loss(self.x, self.y))



    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)


        theta_change_total = float('inf') # initalize variable
        while theta_change_total > self.CONVERGENCE_MARGIN:

            # create a minibatch of random indexes of words
            minibatch = np.random.randint(0,self.DATAPOINTS,self.MINIBATCH_SIZE)

            # get our self.gradient based on minibatrch
            self.compute_gradient_minibatch(minibatch)

            # Uppdatera alla vikter efter varje gradient, inget 0 problem
            self.theta = self.theta - self.LEARNING_RATE * self.gradient

            # Euclidian distance of how much theta changes, close to 0 means we have converged, squared sum of all
            theta_change_total = np.sum(self.gradient ** 2)

            # Plot the loss
            self.update_plot(self.loss(self.x, self.y))




    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        # While we are above the margin
        theta_change_total = float('inf')
        while theta_change_total > self.CONVERGENCE_MARGIN:

            self.compute_gradient_for_all() # compute gradient.

            # Uppdatera alla vikter efter varje gradient, inget 0 problem
            self.theta = self.theta - self.LEARNING_RATE*self.gradient

            # Euclidian distance of how much theta changes, close to 0 means we have converged, squared sum of all
            theta_change_total = np.sum(self.gradient ** 2)

            # Plot the loss
            self.update_plot(self.loss(self.x, self.y))


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, 1+max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
