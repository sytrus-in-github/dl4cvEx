from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable





class Solver(object):

    default_adam_args = {"lr": 1e-4,

                         "betas": (0.9, 0.999),

                         "eps": 1e-8,

                         "weight_decay": 0.0}



    def __init__(self, optim=torch.optim.Adam, optim_args={},

                 loss_func=torch.nn.CrossEntropyLoss):

        optim_args_merged = self.default_adam_args.copy()

        optim_args_merged.update(optim_args)

        self.optim_args = optim_args_merged

        self.optim = optim

        self.loss_func = loss_func()



        self._reset_histories()



    def _reset_histories(self):

        """

        Resets train and val histories for the accuracy and the loss.

        """

        self.train_loss_history = []

        self.train_acc_history = []

        self.val_acc_history = []
        
        self.val_loss_history = []



    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=100, in_cuda=True):

        """

        Train a given model with the provided data.



        Inputs:

        - model: model object initialized from a torch.nn.Module

        - train_loader: train data in torch.utils.data.DataLoader

        - val_loader: val data in torch.utils.data.DataLoader

        - num_epochs: total number of training epochs

        - log_nth: log training accuracy and loss every nth iteration

        """

        if in_cuda is True:
            model.cuda()
        
        optim = self.optim(model.parameters(), **self.optim_args)

        self._reset_histories()

        iter_per_epoch = len(train_loader)



        print 'START TRAIN.'

        ############################################################################

        # TODO:                                                                    #

        # Write your own personal training method for our solver. In Each epoch    #

        # iter_per_epoch shuffled training batches are processed. The loss for     #

        # each batch is stored in self.train_loss_history. Every log_nth iteration #

        # the loss is logged. After one epoch the training accuracy of the last    #

        # mini batch is logged and stored in self.train_acc_history.               #

        # We validate at the end of each epoch, log the result and store the       #

        # accuracy of the entire validation set in self.val_acc_history.           #

        #

        # Your logging should like something like:                                 #

        #   ...                                                                    #

        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #

        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #

        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #

        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #

        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #

        #   ...                                                                    #

        ############################################################################

        for e in xrange(num_epochs):
            model.train()
#            total = 0
#            correct = 0
            train_losss = []
            train_accs = []
            for batch_idx, (data, target) in enumerate(train_loader):
#                print data.size()
                if in_cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optim.zero_grad()
                output = model(data)
#                print target.size()
                loss = self.loss_func(output, target)
                loss.backward()
                train_losss.append(loss.data.cpu().numpy())
                optim.step()
                _, preds = torch.max(output, 1)
                labels_mask = target >= 0
                train_accs.append(np.mean((preds == target)[labels_mask].data.cpu().numpy()))
#                pred = output.data.max(1)[1]
#                total += len(target.view(-1))
#                correct += pred.eq(target.data).sum()
                if batch_idx % log_nth == 0:
                    print '[Iteration {}/{}] TRAIN loss: {}'.format(batch_idx ,iter_per_epoch, np.mean(train_losss))
            train_acc = np.mean(train_accs)
            train_loss = np.mean(train_losss)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            model.eval()
#            total = 0
#            correct = 0
            val_losss = []
            val_accs = []
            for batch_idx, (data, target) in enumerate(val_loader):
#                data = pad_32x(data)
                if in_cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data, volatile=True), Variable(target) 
                output = model(data)
#                output = output.narrow(2, 0, target.size(1)).narrow(3, 0, target.size(2))
                loss = self.loss_func(output, target)
                val_losss.append(loss.data.cpu().numpy())
                _, preds = torch.max(output, 1)
                labels_mask = target >= 0
                val_accs.append(np.mean((preds == target)[labels_mask].data.cpu().numpy()))
#                pred = output.data.max(1)[1]
#                total += len(target.view(-1))
#                correct += pred.eq(target.data).sum()
#            val_acc = correct * 1. / total
#            val_loss = loss.data[0]
            val_acc = np.mean(val_accs)
            val_loss = np.mean(val_losss)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            print '[Epoch {}/{}] TRAIN acc/loss: {}/{}'.format(e+1, num_epochs, train_acc, train_loss)
            print '[Epoch {}/{}] VAL acc/loss: {}/{}'.format(e+1, num_epochs, val_acc, val_loss)

        ############################################################################

        #                             END OF YOUR CODE                             #

        ############################################################################

        print 'FINISH.'

