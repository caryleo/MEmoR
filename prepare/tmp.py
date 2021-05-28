class DenseGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=(2.0/(self.in_channels+self.out_channels))**0.5)
        nn.init.constant_(self.bias, 0)
        # glorot(self.weight)
        # zeros(self.bias)

    def forward(self, x, adj):

        out = torch.matmul(x, self.weight)
        # deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        if self.bias is not None:
            out = out + self.bias
        # adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        return out