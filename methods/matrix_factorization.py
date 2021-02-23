from .utils import Accumulator

from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()

class MF(nn.Block):
    def __init__(self, num_factors : int, num_users : int, num_items : int, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim = num_users, output_dim = num_factors)
        self.Q = nn.Embedding(input_dim = num_items, output_dim = num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id : int, item_id : int):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis = 1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()

def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        R_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        R_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels = R_ui, preds = R_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))

def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator=None, **kwargs):
    rmse_list = []

    for epoch in range(num_epochs):
        metric, l =  Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))

            train_feat = input_data[0:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
        if len(kwargs) > 0:
            test_rmse = evaluator(net, test_iter, kwargs["inter_mat"], devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l/(i+1)
        rmse_list.append(test_rmse)

    print(f"Final train loss {metric[0]/metric[1]:.3f}, Final test RMSE {test_rmse:.3f}")
    return rmse_list
