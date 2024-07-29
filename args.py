import argparse

def arguments():
    parser = argparse.ArgumentParser ()
    #Datasets
    parser.add_argument ('--gpu', type=int, default=1, help='which gpu to use')
    parser.add_argument ('--dataset_dir', type=str, default='./Datasets')
    parser.add_argument ('--dataset_name', type=str, choices=['HuaNan', 'PeMS','PeMS_F'], default="HuaNan")
    parser.add_argument ('--traffic_state', type=str, choices=['Speed', 'Flow'], default="Flow")
    parser.add_argument ('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument ('--seq_len', type=int, default=12, help='Time dimension')
    parser.add_argument ('--input_dim', type=int, default=1)
    parser.add_argument ('--output_dim', type=int, default=1)
    parser.add_argument ('--lane_num_nodes', type=int, default=72,help='PeMS:40,PeMS_F:43,HuaNan:72',)
    parser.add_argument ('--road_num_nodes', type=int, choices=[8, 18], default=18,help='PeMS:8,HuaNan:18')
    parser.add_argument ('--road_lane_count',  default=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4], help='pems43 [5, 6, 5, 6, 5, 6, 5, 5],pems40 [5,5,5,5,5,5,5,5],huanan [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]')


    parser.add_argument ('--batch_first', type=eval, choices=[True, False], default='True')
    parser.add_argument ('--bias', type=eval, choices=[True, False], default='True')
    parser.add_argument ('--return_all_layers', type=eval, choices=[True, False], default='False')
    parser.add_argument ('--is_graph_based', type=eval, choices=[True, False], default='False')
    parser.add_argument ('--base_lr', type=float, default=0.01)
    parser.add_argument ('--epsilon', type=float, default=1.0e-3)
    parser.add_argument ('--steps', type=eval, default=[20, 30, 40, 50])
    parser.add_argument ('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument ('--epochs', type=int, default=100)
    parser.add_argument ('--max_grad_norm', type=int, default=5)
    parser.add_argument ('--patience', type=int, default=10)

    parser.add_argument ('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument ('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument ('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument ('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument ('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument ('--factor', type=int, default=1, help='attn factor')
    parser.add_argument ('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument ('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument ('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument ('--activation', type=str, default='gelu', help='activation')
    parser.add_argument ('--head_dropout', type=float, default=0.1, help='head dropout')

    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument ('--graph_based', type=bool, default=False, help='graph_based; True  False ')
    parser.add_argument ('--patch_len', type=int, default=4, help='patch length')
    parser.add_argument ('--stride', type=int, default=2, help='stride')
    parser.add_argument ('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument ('--isNoise', type=bool, default=False, help='Whether to add noise to the training data')
    parser.add_argument ('--Noise_ratio', type=float, default=0.025, help='Add random noise ratio')
    parser.add_argument ('--diffusion_step', type=int, default=10)


    args = parser.parse_args ()
    return args
