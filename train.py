from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

from utils import load_data, dotdict, train_test_split, make_data_loader
from model import MatrixFactorization

def main(args):
    """ Train collaborative filtering """ 
    # Clean slate
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    R = load_data(args.dataset_str)
    
    # Split into training/test
    n_users, n_items = R.shape
    R_train, R_test = train_test_split(R, test_size=args.test_size)

    # Make data handlers
    train_loader = make_data_loader(R_train, batch_size=args.batch_size,)
    test_loader = make_data_loader(R_test, batch_size=args.batch_size,)
    
    # Define model, loss, optimizer
    model = MatrixFactorization(n_users, n_items, n_factors=args.n_latent)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SparseAdam(model.parameters(), lr = args.lr)
    

    for i in range(args.num_epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        test_loss = 0.0

        for j, (data) in enumerate(train_loader):
            # Turn data into variables
            row, col, rating = data
            rating = Variable(rating.float())
            row = Variable(row.long())
            col = Variable(col.long())
            
            # Make prediction
            prediction = model(row, col)
            loss = loss_func(prediction, rating)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Backpropagate
            loss.backward()
            
            # Update the parameters
            optimizer.step()

            # Save loss
            train_loss += loss.data[0]
        
        # -----------------------
        # Print every epoch
        for test_data in test_loader:
            row, col, rating = test_data
            rating = Variable(rating.float())
            row = Variable(row.long())
            col = Variable(col.long())

            prediction = model(row, col)
            loss = loss_func(prediction, rating)
            test_loss += loss.data[0]

        # print statistics
        print('epoch: {}, train_loss: {}, test_loss: {}'.format(
        i + 1, train_loss/len(train_loader), test_loss/len(test_loader)))


    print('Finished Training!')
    

if __name__ == '__main__':

    args = dotdict()
    args.dataset_str = 'delicious'
    # args.dataset_str = 'lastfm'
    args.seed        = 0
    args.test_size   = 0.2
    args.n_latent    = 40
    args.dropout     = 0.5
    args.num_epochs  = 5
    args.batch_size  = 1000
    args.lr          = 1e-1
    args.momentum    = 0.9

    main(args)
    #
