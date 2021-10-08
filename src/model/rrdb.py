from model import common_rrdb as common
#from config import get_config

from torch import nn
from torch.nn import functional

def make_model(args, parent=False):
    return RRDB(args)

class RDBlock(nn.Module):

    def __init__(
            self, n_feats, gf, drep=5, res_scale=0.2, conv=common.default_conv):

        super(RDBlock, self).__init__()
        self.n_feats = n_feats
        self.n_max = n_feats + (drep - 1) * gf
        self.gf = gf
        self.res_scale = res_scale
        m = [conv(n_feats + i * gf, gf, 3) for i in range(drep - 1)]
        m.append(conv(self.n_max, n_feats, 3))
        self.convs = nn.ModuleList(m)

    def forward(self, x):
        '''
        We will not use torch.cat since it consumes GPU memory lot.
        '''
        b, _, h, w = x.size()
        c = self.n_feats
        buf = x.new_empty(b, self.n_max, h, w)
        buf[:, :c, :, :] = x
        for i, conv in enumerate(self.convs):
            x_inter = conv(buf[:, :c, :, :])
            if i == len(self.convs) - 1:
                return x + self.res_scale * x_inter
            else:
                x_inter = functional.leaky_relu(
                    x_inter, negative_slope=0.2, inplace=True
                )
                buf[:, c:c + self.gf, :, :] = x_inter
                c += self.gf


class RRDBlock(nn.Sequential):

    def __init__(
            self, n_feats, gf, rep=3, res_scale=0.2, conv=common.default_conv):

        self.res_scale = res_scale
        args = [n_feats, gf]
        kwargs = {'res_scale': res_scale, 'conv': conv}
        m = [RDBlock(*args, **kwargs) for _ in range(rep)]
        super().__init__(*m)

    def forward(self, x):
        x = x + self.res_scale * super().forward(x)
        return x


class RRDB(nn.Module):
    '''
    RRDB model

    Note:
        From
        "ESRGAN"
        See for more detail.
    '''

    def __init__(
            self, args, scale=4, depth=23, n_colors=3, n_feats=64,
            gf=32, res_scale=0.2, conv=common.default_conv):

        super(RRDB, self).__init__()
        scale = int(args.scale)
        self.conv = conv(n_colors, n_feats, 3)
        block = lambda: RRDBlock(n_feats, gf, res_scale=res_scale)
        m = [block() for _ in range(depth)]
        m.append(conv(n_feats, n_feats, 3))
        self.rrdblocks = nn.Sequential(*m)
        self.recon = nn.Sequential(
            common.UpsamplerI(scale, n_feats, algorithm='nearest'),
            conv(n_feats, n_feats, 3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv(n_feats, n_colors, 3)
        )
        self.res_scale = res_scale

    @staticmethod
    def get_kwargs(cfg, conv=common.default_conv):
        parse_list = ['scale', 'n_colors', 'n_feats']
        kwargs = get_config.parse_namespace(cfg, *parse_list)
        kwargs['gf'] = cfg.n_feats // 2
        kwargs['depth'] = 23
        kwargs['res_scale'] = 0.2
        kwargs['conv'] = conv
        return kwargs

    def forward(self, x):
        x = self.conv(x)
        x = x + self.rrdblocks(x)
        x = self.recon(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for k in own_state.keys():
            if k not in state_dict and 'recon' not in k:
                raise RuntimeError(k + ' does not exist!')
            else:
                if k in state_dict:
                    own_state[k] = state_dict[k]

        super().load_state_dict(own_state, strict=strict)

# For loading the module
REPRESENTATIVE = RRDB

