from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()

class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout = 0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation="sigmoid", use_bias = True)
        self.decoder = nn.Dense(num_users, use_bias = True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():
            return pred*np.sign(input)
        else:
            return pred

def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                    / np.sum(np.sign(test_data)))
    return float(rmse)