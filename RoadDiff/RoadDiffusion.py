import torch
import torch.nn as nn
import torch.optim as optim

# https://github.com/ts-kim/RevIN/blob/master/RevIN.py
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps*self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SelfAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(e, self.a).squeeze(2))

        attention = nn.functional.softmax(e, dim=1)
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return nn.functional.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(AttentionMLP, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            SelfAttentionLayer(input_dim, hidden_dim, dropout=0.5, alpha=0.2, concat=True)
            for _ in range(num_heads)
        ])
        self.out_att = SelfAttentionLayer(hidden_dim * num_heads, output_dim, dropout=0.5, alpha=0.2, concat=False)

    def forward(self, x):
        x = torch.cat([att(x) for att in self.attention_heads], dim=1)
        x = self.out_att(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_heads):
        super(Encoder, self).__init__()
        self.att_mlp = AttentionMLP(input_dim, hidden_dim, latent_dim, num_heads)

    def forward(self, x):
        batch_size, num_nodes, time_window_size = x.size()
        h = self.att_mlp(x.reshape(-1, time_window_size))
        return h.view(batch_size, num_nodes, -1), h.view(batch_size, num_nodes, -1)  # 返回均值和对数方差

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_heads,input_num,output_num, output_dim):
        super(Decoder, self).__init__()
        self.att_mlp = AttentionMLP(latent_dim, hidden_dim, output_dim, num_heads)
        self.fc1 = nn.Linear (output_dim * input_num, output_dim*output_num)
        self.output_num = output_num
        self.output_dim = output_dim

    def forward(self, z):
        batch_size, num_nodes, latent_dim = z.size()
        h = self.att_mlp(z.view(-1, latent_dim))
        out = self.fc1(h.view(batch_size , self.output_dim* num_nodes))
        return out.view(batch_size, self.output_num, -1)

class DenoisingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        batch_size, num_nodes, time_window_size = x.size()
        x = x.view(batch_size * num_nodes, time_window_size)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc3(h)
        sigma = torch.sigmoid(self.fc3(h))  # 确保标准差为正
        return mu.view(batch_size, num_nodes, -1), sigma.view(batch_size, num_nodes, -1)

class RoadDiffusion(nn.Module):
    def __init__(self, time_window_size, num_road_nodes, num_lane_nodes,timesteps,target_slice=slice(0,None,None)):
        super(RoadDiffusion, self).__init__()
        self.encoder = Encoder(input_dim=time_window_size, hidden_dim=50, latent_dim=30, num_heads=3)
        self.decoder = Decoder(latent_dim=30, hidden_dim=50, num_heads=3, input_num = num_road_nodes,output_num = num_lane_nodes,output_dim=time_window_size)
        self.denoising_network = DenoisingNetwork(input_dim=time_window_size, hidden_dim=50)
        input_shape = (time_window_size, num_road_nodes)
        self.target_slice = target_slice
        self.rev_norm = RevIN (input_shape[0])
        self.num_lane_nodes = num_lane_nodes
        self.timesteps = timesteps

    def forward(self, x_R):
        x_R = self.rev_norm(x_R, 'norm')
        z_mean, z_logvar = self.encoder(x_R)
        z = self.reparameterize(z_mean, z_logvar)
        initial_X_L = self.decoder(z)
        X_T = self.forward_diffusion(initial_X_L, self.timesteps)
        predicted_X_L = self.reverse_diffusion(X_T, self.timesteps)
        predicted_X_L = self.rev_norm(predicted_X_L,'denorm', self.target_slice)
        return predicted_X_L

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward_diffusion(self, X_0, timesteps):
        alpha = torch.linspace(0.1, 0.9, timesteps)
        X_t = X_0
        for t in range(timesteps):
            noise = torch.randn_like(X_t)
            X_t = torch.sqrt(alpha[t]) * X_t + torch.sqrt(1 - alpha[t]) * noise
        return X_t

    def reverse_diffusion(self, X_T, timesteps):
        X_t = X_T
        for t in reversed(range(timesteps)):
            mu, sigma = self.denoising_network(X_t, t)
            noise = torch.randn_like(X_t)
            X_t = mu + sigma * noise
        return X_t