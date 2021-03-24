# Import packages used throughout
import torch
from torch import nn, optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import import_data 
from methods import utils, losses, matrix_factorization #, NeuMF, Caser

def main():
    # Import data
    names = ["user_id", "item_id", "rating", "timestamp"]
    data = pd.read_csv("data/u.data", delimiter='\t', names = names, engine = "python")
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]

    # Obtain the data
    num_users, num_items, train_iter, test_iter = import_data.split_and_load_ml100k(data,
        num_users, num_items, test_ratio = 0.1, batch_size=512)

    # Define evaluator
    def evaluator(net, test_iter):
        rmse_list = []
        for idx, (users, items, ratings) in enumerate(test_iter):
            r_hat = net(users, items)
            rmse_value = torch.sqrt(((r_hat - ratings)**2).mean())
            rmse_list.append(float(rmse_value))

        return float(np.mean(np.array(rmse_list)))

    # Prepare model
    lr, num_epochs, wd = 0.002, 20, 1e-5

    mf_net = matrix_factorization.MF(30, num_users, num_items)
    optimizer = optim.Adam(mf_net.parameters(), lr = lr, weight_decay = wd)

    # Train and evaluate model
    rmse_list = []
    for epoch in range(num_epochs):
        accumulator, l = utils.Accumulator(2), 0.

        # Train each batch
        mf_net.train()
        for i, (users, items, ratings) in enumerate(train_iter):
            optimizer.zero_grad()

            predictions = mf_net(users, items)
            output = ((predictions - ratings)**2).mean()
            output.backward()
            optimizer.step()
            accumulator.add(output, users.shape[0])
        
        # Evaluate
        mf_net.eval()
        rmse = evaluator(mf_net, test_iter)
        rmse_list.append(rmse)

        print(f"Epoch {epoch}:\n\tloss = {accumulator[0]/accumulator[1]}\n\trmse = {rmse}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()