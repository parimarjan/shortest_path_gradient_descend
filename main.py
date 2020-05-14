import networkx as nx
import torch
import pdb
from utils import *
from tensorflow import summary as tf_summary
import time
import sys

MIN_VAL = 10
MAX_VAL = 100
MAX_COST = MAX_VAL*2*DIFF_ALPHA
NUM_EPOCHS = 1000
EVAL_EPOCH = 1
NUM_SAMPLES = 25
LR = 0.01
USE_FLOW = True
# binary tree parameters
WIDTH = 2
HEIGHT = 3
WEIGHT_INIT = 0
NORMALIZE_FEATURES = True

if USE_FLOW is None:
    prefix = "true"
elif USE_FLOW == True:
    prefix = "flow"
else:
    prefix = "mse"

if len(sys.argv) > 1:
    prefix = sys.argv[1] + prefix

prefix += "w" + str(WIDTH) + "h" + str(HEIGHT)

log_dir = "./tfboard/" + prefix + str(time.time())

tf_summary_writer = tf_summary.create_file_writer(log_dir)

train_samples = gen_dataset(NUM_SAMPLES, WIDTH,HEIGHT, min_val=MIN_VAL, max_val=MAX_VAL)
print("generated dataset!")

if NORMALIZE_FEATURES:
    trainX, trainY = get_training_features(train_samples, MIN_VAL, MAX_VAL, MAX_COST)
else:
    trainX, trainY = get_training_features(train_samples, None, None, None)

if USE_FLOW is None:
    net = None
    optimizer = None
else:
    net = SimpleRegression(len(trainX[0][0]), None, hidden_layer_size=5,
            n_output=1, num_hidden_layers=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR,
        amsgrad=False)
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR)

if WEIGHT_INIT:
    print(net)
    new_weights = {}
    for key, weights in net.state_dict().items():
        print(key, len(weights))
        # new_weights[key] = torch.zeros(weights.shape)
        sh = weights.shape
        # print(sh[0], sh[1])
        if WEIGHT_INIT == 1:
            if len(sh) == 2:
                new_weights[key] = \
                        to_variable(np.random.rand(sh[0], sh[1]))
            else:
                new_weights[key] = \
                        to_variable(np.random.rand(sh[0]))
        elif WEIGHT_INIT == 2:
                new_weights[key] = torch.zeros(weights.shape)
        elif WEIGHT_INIT == 4:
                new_weights[key] = torch.ones(weights.shape)
        elif WEIGHT_INIT == 3:
                SCALE = random.random()
                print(SCALE)
                if len(sh) == 2:
                    new_weights[key] = \
                            to_variable(SCALE*np.random.rand(sh[0], sh[1]))
                else:
                    new_weights[key] = \
                            to_variable(SCALE*np.random.rand(sh[0]))

                # new_weights[key] = torch.zeros(weights.shape)

        # print(weights.shape)
        # pdb.set_trace()

    # if "bias" not in key:
        # new_weights[key][-1][-1] = 0.00
    net.load_state_dict(new_weights)
    print("state dict updated")

if USE_FLOW == True:
    loss_fn = flow_loss
elif USE_FLOW == 3:
    loss_fn1 = flow_loss
    loss_fn2 = torch.nn.MSELoss()
else:
    loss_fn = torch.nn.MSELoss()

true_fl = None
for i in range(NUM_EPOCHS):
    if i % EVAL_EPOCH == 0:
        mse_loss = eval_loss(torch.nn.MSELoss(reduce=None), net, trainX, trainY, None)
        fl = eval_loss(flow_loss, net, trainX, trainY, train_samples)
        spl = eval_loss(sp_loss, net, trainX, trainY, train_samples)

        print("{}, mse: {}, flow loss: {}, spl: {}".format(i, mse_loss, fl, spl))
        # print("{}, mse: {}, flow loss: {}, true_fl: {}, spl: {}".format(
            # i, mse_loss, fl, true_fl, spl))
        with tf_summary_writer.as_default():
            tf_summary.scalar("mse_loss", mse_loss, step=i)
            tf_summary.scalar("flow_loss", fl, step=i)
            tf_summary.scalar("sp_loss", spl, step=i)

    if USE_FLOW is None:
        continue
    tidx = random.randint(0, len(trainX)-1)
    xbatch = trainX[tidx]
    ybatch = trainY[tidx]
    sample = train_samples[tidx]
    pred = net(xbatch).squeeze(1)
    assert pred.shape == ybatch.shape
    if USE_FLOW == 3:
        loss = loss_fn1(sample, pred, ybatch)
        # loss = loss + 0.01*loss_fn2(pred, ybatch)
        loss = loss + 0.1*loss_fn2(pred[1:2], ybatch[1:2])

        # pdb.set_trace()
        # print(loss)
    elif USE_FLOW:
        loss = loss_fn(sample, pred, ybatch)
    else:
        loss = loss_fn(pred, ybatch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# for i, xbatch in enumerate(trainX):
    # for j
