# import argparse
# import yaml
import numpy as np
import os
import torch
import time
from RoadDiff.RoadDiffusion import RoadDiffusion
from args import arguments

args = arguments()
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DataLoader (object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        print (len (xs), batch_size)
        if pad_with_last_sample:
            num_padding = (batch_size - (len (xs) % batch_size)) % batch_size
            x_padding = np.repeat (xs[-1:], num_padding, axis=0)
            y_padding = np.repeat (ys[-1:], num_padding, axis=0)
            xs = np.concatenate ([xs, x_padding], axis=0)
            ys = np.concatenate ([ys, y_padding], axis=0)
        self.size = len (xs)
        self.num_batch = int (self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation (self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min (self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper ()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir,dataset_name,seq_len,traffic_state, batch_size,data_type):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, dataset_name,traffic_state,data_type,category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['x_' + category] = data['x_' + category][:, 0:int (seq_len), :, :]
        data['y_' + category] = cat_data['y']
        data['y_' + category] = data['y_' + category][:,0:int(seq_len),:,:]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


def get_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
             y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = torch.from_numpy (x).float ()
    y = torch.from_numpy (y).float ()
    # print ("X: {}".format (x.size ()))
    # print ("y: {}".format (y.size ()))
    x = x.permute (1, 0, 2, 3)
    y = y.permute (1, 0, 2, 3)
    return x, y


def get_x_y_in_correct_dims(x, y,type):
    """
    :param x: shape (seq_len, batch_size, num_sensor, input_dim)
    :param y: shape (horizon, batch_size, num_sensor, input_dim)
    :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
            y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    if type == 'lane':
        batch_size = x.size (1)
        x = x.view (args.seq_len, args.batch_size, args.lane_num_nodes, args.input_dim)
        y = y[..., :args.output_dim].view (args.seq_len, batch_size,
                                           y.size (2), args.output_dim)
    else:
        batch_size = x.size (1)
        x = x.view (args.seq_len, args.batch_size, args.road_num_nodes, args.input_dim)
        y = y[..., :args.output_dim].view (args.seq_len, batch_size,
                                           y.size (2), args.output_dim)

    return x, y


def prepare_data(x, y,type):
    x, y = get_x_y (x, y)
    x, y = get_x_y_in_correct_dims (x, y,type)
    return x.to (device), y.to (device)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float ()
    mask /= mask.mean ()
    loss = torch.abs (y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean ()


def compute_loss(y_true, y_predicted):
    y_true = standard_scaler_lane.inverse_transform (y_true)
    y_predicted = standard_scaler_lane.inverse_transform (y_predicted)
    return masked_mae_loss (y_predicted, y_true)


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where (torch.isnan (loss), torch.zeros_like (loss), loss)
    return torch.mean (loss)


def calc_metrics(preds, labels, null_val=0.):
    preds = torch.from_numpy (preds)
    labels = torch.from_numpy (labels)
    if np.isnan (null_val):
        mask = ~torch.isnan (labels)
    else:
        mask = (labels != null_val)
    mask = mask.float ()
    mask /= torch.mean (mask)
    mask = torch.where (torch.isnan (mask), torch.zeros_like (mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs (preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna (l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt (mse)
    return mae, mape, rmse


def evaluate(model, dataset='val', batches_seen=0):
    """
    Computes mean L1Loss
    :return: mean L1Loss
    """
    with torch.no_grad ():
        model = model.eval ()

        val_iterator = data_lane['{}_loader'.format (dataset)].get_iterator ()
        val_iterator_road = data_road['{}_loader'.format (dataset)].get_iterator ()
        losses = []

        y_truths = []
        y_preds = []

        for _, ((x, y), (x_road, y_road)) in enumerate (zip (val_iterator, val_iterator_road)):
            x, y = prepare_data (x, y,'lane')
            x_road, y_road = prepare_data (x_road, y_road,'road') # H,B,N,D
            x = x.squeeze (3).permute (1,2,0)
            x_road = x_road.squeeze (3).permute (1,2,0)
            output = model (x_road)
            loss = compute_loss (x, output)
            losses.append (loss.item ())
            y_truths.append (x.cpu ())
            y_preds.append (output.cpu ())


        mean_loss = np.mean (losses)
        y_preds = np.concatenate (y_preds, axis=1)
        y_truths = np.concatenate (y_truths, axis=1)  # concatenate on batch dimension


        y_truths_scaled = []
        y_preds_scaled = []

        for t in range (y_preds.shape[0]):
            y_truth = standard_scaler_lane.inverse_transform (y_truths[t])
            y_pred = standard_scaler_lane.inverse_transform (y_preds[t])
            y_truths_scaled.append (y_truth)
            y_preds_scaled.append (y_pred)


        y_preds_scaled = np.array (y_preds_scaled)
        y_truths_scaled = np.array (y_truths_scaled)
        mae, mape, rmse = calc_metrics (y_preds_scaled, y_truths_scaled, null_val=0.0)
        print ('lane  mae:', mae.item (), 'mape:', mape.item (), 'rmse:', rmse.item ())
        return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}


def set_na_randomly(tensor, k):
    """
    在PyTorch张量的seq_len维度上随机设置元素为NA。

    :param tensor: 输入的PyTorch张量，形状为(batch_size, seq_len, input_dim, num_node, out_dim)
    :param k: 比例，指定在seq_len维度上设置为NA的元素比例
    """
    # 确定每个batch和node维度上需要设置为NA的元素数量
    seq_len = tensor.shape[1]
    num_to_select = int (seq_len * k)

    # 遍历张量的batch_size, input_dim, num_node和out_dim维度
    for i in range (tensor.shape[0]):  # batch_size维度
        for j in range (tensor.shape[2]):  # input_dim维度
            for m in range (tensor.shape[3]):  # num_node维度
                for n in range (tensor.shape[4]):  # out_dim维度
                    # 随机选择索引
                    indices = torch.randperm (seq_len)[:num_to_select]
                    # 设置为torch.nan
                    tensor[i, indices, j, m, n] = 1
    return tensor


if __name__ == '__main__':

    data_lane = load_dataset (args.dataset_dir,args.dataset_name,args.seq_len,args.traffic_state, args.batch_size,'Lane')
    data_road = load_dataset (args.dataset_dir, args.dataset_name, args.seq_len, args.traffic_state, args.batch_size,'Road')
    standard_scaler_lane = data_lane['scaler']
    standard_scaler_road = data_road['scaler']

    model = RoadDiffusion(args.seq_len, args.road_num_nodes, args.lane_num_nodes,args.diffusion_step)
    model.to (device)

    """
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    optimizer = torch.optim.Adam (model.parameters (), lr=args.base_lr, eps=args.epsilon)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR (optimizer, milestones=args.steps,
                                                         gamma=args.lr_decay_ratio)

    num_batches = data_lane['train_loader'].num_batch
    print ('num params:', count_parameters (model))

    for epoch_num in range (0, args.epochs):

        model = model.train ()

        train_iterator = data_lane['train_loader'].get_iterator ()
        train_iterator_road = data_road['train_loader'].get_iterator ()
        losses = []
        for _, ((x, y),(x_road,y_road)) in enumerate (zip(train_iterator,train_iterator_road)):
            start_time = time.time ()
            optimizer.zero_grad ()
            x, y = prepare_data (x, y,'lane')
            x_road, y_road = prepare_data (x_road, y_road,'road') # H,B,N,D
            x = x.squeeze (3).permute (1,2,0)
            x_road = x_road.squeeze (3).permute (1,2,0)
            output = model (x_road)
            loss = compute_loss (x, output)
            losses.append (loss.item ())
            loss.backward ()

            # gradient clipping - this does it in place
            torch.nn.utils.clip_grad_norm_ (model.parameters (), args.max_grad_norm)

            optimizer.step ()
        print ("epoch complete")
        lr_scheduler.step ()
        print ("evaluating now!")

        val_loss, _ = evaluate (model, dataset='val')


    test_loss, _ = evaluate (model, dataset='test')
