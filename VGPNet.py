import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import batch_episym


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super().__init__()
        self.pre = pre
        self.right = nn.Sequential(nn.Conv2d(inchannel, outchannel, (1, 1)))
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre else x
        return torch.relu(self.left(x) + x1)


def batch_symeig(X):
    device = X.device
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for bi in range(X.shape[0]):
        _, v = torch.linalg.eigh(X[bi].squeeze(), UPLO='U')
        bv[bi] = v
    return bv.to(device)


def weighted_8points(x_in, logits):
    mask    = torch.sigmoid(logits[:, 0, :, 0])
    weights = torch.exp(logits[:, 1, :, 0]) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in  = x_in.squeeze(1)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0],            xx[:, 1],            torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()

    wX   = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX  = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)
    v    = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx    = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20, idx=None):
    batch_size, num_points = x.shape[0], x.shape[2]
    x = x.view(batch_size, -1, num_points)
    idx_out = knn(x, k=k) if idx is None else idx
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx_out + idx_base).view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        super().__init__()
        if not out_channels:
            out_channels = channels
        self.shot_cut = nn.Conv2d(channels, out_channels, 1) if out_channels != channels else None
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, 1),
            trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, 1))
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        return out + (self.shot_cut(x) if self.shot_cut else x)


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        super().__init__()
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, 1))

    def forward(self, x):
        S = torch.softmax(self.conv(x), dim=2).squeeze(3)
        return torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        super().__init__()
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, 1))

    def forward(self, x_up, x_down):
        S = torch.softmax(self.conv(x_up), dim=1).squeeze(3)
        return torch.matmul(x_down.squeeze(3), S).unsqueeze(3)


class OABlock(nn.Module):
    def __init__(self, net_channels, depth=6, clusters=250):
        super().__init__()
        channels = net_channels
        self.layer_num = depth
        self.down1 = diff_pool(channels, clusters)
        self.l2 = nn.Sequential(*[OAFilter(channels, clusters) for _ in range(depth // 2)])
        self.up1 = diff_unpool(channels, clusters)
        self.output = nn.Conv2d(channels, 1, 1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, data):
        x_down = self.down1(data)
        x2     = self.l2(x_down)
        x_up   = self.up1(data, x2)
        return self.shot_cut(torch.cat([data, x_up], dim=1))


class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True))


class MLPs(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, 1))

    def forward(self, x):
        return self.conv(x)


class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter = channels // r
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter, 1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1),
            nn.BatchNorm2d(channels))
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter, 1),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1),
            nn.BatchNorm2d(channels))
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.tensor(1.5))

    def forward(self, x, residual):
        xa  = x + residual
        wei = self.sigmoid(self.local_att(xa) + self.global_att(xa))
        s   = torch.clamp(self.scale, 0.5, 3.0)
        return s * x * wei + s * residual * (1 - wei)


class GNN(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super().__init__()
        assert knn_num in (9, 6)
        self.knn_num    = knn_num
        self.in_channel = in_channel
        self.mlp1    = MLPs(2 * in_channel, 2 * in_channel)
        self.change1 = MLPs(2 * in_channel, in_channel)
        self.change2 = MLPs(2 * in_channel, in_channel)
        self.aff     = AFF(in_channel, 4)
        if knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel * 2, in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(in_channel * 2), nn.ReLU(inplace=True),
                nn.Conv2d(in_channel * 2, in_channel * 2, (1, 3)),
                nn.BatchNorm2d(in_channel * 2), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel * 2, in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(in_channel * 2), nn.ReLU(inplace=True),
                nn.Conv2d(in_channel * 2, in_channel * 2, (1, 2)),
                nn.BatchNorm2d(in_channel * 2), nn.ReLU(inplace=True))

    def forward(self, features):
        out = get_graph_feature(features, k=self.knn_num)
        out_an = self.change1(self.conv(out))
        out_max = self.mlp1(out).max(dim=-1, keepdim=False)[0].unsqueeze(3)
        out_max = self.change2(out_max)
        return self.aff(out_max, out_an) + features


class CPT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_A = nn.Conv2d(2, in_channels, 1)
        self.enc_B = nn.Conv2d(2, in_channels, 1)
        self.enc_rel = nn.Sequential(
            nn.Conv2d(3, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1))

        self.q = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                               nn.BatchNorm2d(in_channels), nn.ReLU())
        self.k = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                               nn.BatchNorm2d(in_channels), nn.ReLU())
        self.v = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                               nn.BatchNorm2d(in_channels), nn.ReLU())
        self.temperature  = torch.sqrt(torch.tensor(float(in_channels)))
        self.temperature2 = torch.sqrt(torch.tensor(float(in_channels)))

    def forward(self, feat, x):
        q = self.q(feat).squeeze(3)
        k = self.k(feat).squeeze(3)
        v = self.v(feat).squeeze(3)

        coord_A = x[:, :2]
        coord_B = x[:, 2:4]
        delta   = coord_A - coord_B
        dist    = delta.norm(dim=1, keepdim=True)

        geo_A   = self.enc_A(coord_A)
        geo_B   = self.enc_B(coord_B)
        geo_rel = self.enc_rel(torch.cat([delta, dist], dim=1))
        graph_ctx = (geo_A + geo_B + geo_rel).squeeze(3)

        ctx_pos = torch.matmul(q / self.temperature2, graph_ctx.transpose(1, 2))
        attn    = torch.matmul(q / self.temperature,  k.transpose(1, 2))
        attn    = F.softmax(attn + ctx_pos, dim=-1)
        return torch.matmul(attn, v).unsqueeze(3)


class MSDEA(nn.Module):
    def __init__(self, in_channels=128, reduction=4, use_residual=True):
        super().__init__()
        inter = in_channels // reduction

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels), nn.GELU())
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter, 1),
            nn.BatchNorm2d(inter), nn.GELU(),
            nn.Conv2d(inter, in_channels, 1),
            nn.BatchNorm2d(in_channels))
        self.global_att_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter, 1),
            nn.BatchNorm2d(inter), nn.GELU(),
            nn.Conv2d(inter, in_channels, 1),
            nn.BatchNorm2d(in_channels))
        self.global_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, inter, 1),
            nn.BatchNorm2d(inter), nn.GELU(),
            nn.Conv2d(inter, in_channels, 1),
            nn.BatchNorm2d(in_channels))
        self.global_local_diff = nn.Sequential(
            nn.Conv2d(in_channels, inter, 1),
            nn.BatchNorm2d(inter), nn.GELU(),
            nn.Conv2d(inter, in_channels, 1),
            nn.BatchNorm2d(in_channels))

        self.branch_weights = nn.Parameter(torch.ones(in_channels, 4) * 0.25)
        self.temperature    = nn.Parameter(torch.ones(1))
        self.sigmoid   = nn.Sigmoid()
        self.conv_out  = nn.Conv2d(in_channels, in_channels, 1)
        self.use_residual = use_residual

    def forward(self, x):
        h = self.conv_in(x)
        s1 = self.local_att(h)
        s2 = self.global_att_avg(h)
        s3 = self.global_att_max(h)
        s4 = self.global_local_diff(h - h.mean(dim=2, keepdim=True))

        bw = torch.softmax(
            self.branch_weights / self.temperature.abs().clamp(min=0.1), dim=1)
        w0, w1, w2, w3 = (bw[:, i].view(1, -1, 1, 1) for i in range(4))
        scale = self.sigmoid(w0 * s1 + w1 * s2 + w2 * s3 + w3 * s4)

        out = self.conv_out(h * scale) + h
        if self.use_residual:
            out = out + x
        return out


MSDA = MSDEA


class PCV(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=32, top_k=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.top_k    = top_k

        self.split_A = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))
        self.split_B = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))

        self.geo_encoder_A = nn.Sequential(
            nn.Conv2d(3, feat_dim // 2, 1),
            nn.BatchNorm2d(feat_dim // 2), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, feat_dim, 1))
        self.geo_encoder_B = nn.Sequential(
            nn.Conv2d(3, feat_dim // 2, 1),
            nn.BatchNorm2d(feat_dim // 2), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, feat_dim, 1))

        self.desc_encoder = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.InstanceNorm2d(feat_dim),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.InstanceNorm2d(feat_dim),
            nn.BatchNorm2d(feat_dim))

        self.sim_encoder_main = nn.Sequential(
            nn.Conv2d(2 * feat_dim + 6, hidden_dim, 1),
            nn.InstanceNorm2d(hidden_dim, eps=1e-3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.InstanceNorm2d(hidden_dim, eps=1e-3),
            nn.BatchNorm2d(hidden_dim))
        self.sim_encoder_sc = nn.Conv2d(2 * feat_dim + 6, hidden_dim, 1)

        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.InstanceNorm2d(hidden_dim // 2, eps=1e-3),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid())
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid())
        self.proj128 = nn.Conv2d(hidden_dim, 128, 1)

        self.temp         = nn.Parameter(torch.tensor(1.0))
        self.inject_alpha = nn.Parameter(torch.tensor(0.22))

    def forward(self, feat, coords):
        fA = self.split_A(feat[:, :self.feat_dim])
        fB = self.split_B(feat[:, self.feat_dim:])

        dx   = coords[:, :2] - coords[:, 2:4]
        dist = dx.norm(dim=1, keepdim=True)
        fA   = fA + self.geo_encoder_A(torch.cat([ dx, dist], dim=1))
        fB   = fB + self.geo_encoder_B(torch.cat([-dx, dist], dim=1))

        dA = self.desc_encoder(fA)
        dB = self.desc_encoder(fB)
        B, C, N, _ = dA.shape
        fAf = F.normalize(dA.squeeze(3).transpose(1, 2), dim=-1)
        fBf = F.normalize(dB.squeeze(3).transpose(1, 2), dim=-1)
        S   = torch.bmm(fAf, fBf.transpose(1, 2)) / (self.temp.abs() + 1e-6)

        topA, _ = S.topk(self.top_k, dim=-1)
        topB, _ = S.transpose(1, 2).topk(self.top_k, dim=-1)
        diag = torch.gather(
            S, 2,
            torch.arange(N, device=S.device).view(1, N, 1).expand(B, N, 1)
        ).squeeze(-1)

        dA2B = torch.tanh(diag - topA.mean(-1))
        dB2A = torch.tanh(diag - topB.mean(-1))
        rA2B = (self.top_k - (topA > diag.unsqueeze(-1)).sum(-1)) / self.top_k
        rB2A = (self.top_k - (topB > diag.unsqueeze(-1)).sum(-1)) / self.top_k
        dmaxA2B = torch.tanh(diag - topA.max(-1)[0])
        dmaxB2A = torch.tanh(diag - topB.max(-1)[0])

        fuse = torch.cat([
            dA, dB,
            dA2B.unsqueeze(1).unsqueeze(3),    dB2A.unsqueeze(1).unsqueeze(3),
            rA2B.unsqueeze(1).unsqueeze(3),    rB2A.unsqueeze(1).unsqueeze(3),
            dmaxA2B.unsqueeze(1).unsqueeze(3), dmaxB2A.unsqueeze(1).unsqueeze(3)
        ], dim=1)
        sf = torch.relu(self.sim_encoder_main(fuse) + self.sim_encoder_sc(fuse))
        sf = sf * self.attention(sf)

        return self.predictor(sf), self.proj128(sf), self.inject_alpha


class CGR(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gcn = GCN_Block(channels)
        self.cpt = CPT(channels, channels)
        self.ln1 = nn.LayerNorm(channels, eps=1e-6)
        self.ffn = MSDA(in_channels=channels)
        self.ln2 = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, feature, weight, position_feature, consist_prob=None):
        B, _, N, _ = feature.shape
        with torch.no_grad():
            w_ = torch.relu(torch.tanh(weight)).unsqueeze(-1)
            A  = torch.bmm(w_.transpose(1, 2), w_)
            if consist_prob is not None:
                cp  = consist_prob.squeeze(1).squeeze(-1)
                cp_ = torch.relu(torch.tanh(cp)).unsqueeze(-1)
                A   = A * torch.bmm(cp_.transpose(1, 2), cp_) + A * 0.1
            A  = A + torch.eye(N, device=feature.device).unsqueeze(0)
            D  = torch.diag_embed((1.0 / A.sum(-1)) ** 0.5)
            L  = D @ A @ D
        out_tmp = feature.squeeze(-1).transpose(1, 2)
        out_gcn = torch.bmm(L, out_tmp).unsqueeze(-1).transpose(1, 2).contiguous()
        out_gcn = self.gcn.conv(out_gcn) + feature

        cpt_out = self.cpt(out_gcn, position_feature) + out_gcn
        cpt_ln  = self.ln1(cpt_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        ffn_out = self.ffn(cpt_ln)
        out_ln  = self.ln2(ffn_out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = cpt_ln + out_ln
        return out, out_gcn


class CSP(nn.Module):
    def __init__(self, channels=128, num_keys=3):
        super().__init__()
        self.substage_att = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            ) for _ in range(num_keys)
        ])
        self.conv_adjust = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, key_outs):
        F1, F2, F3 = key_outs
        diff = (F3 - F1).detach()
        w1 = self.substage_att[0](torch.cat([F1,  diff], dim=1))
        w2 = self.substage_att[1](torch.cat([F2,  diff], dim=1))
        w3 = self.substage_att[2](torch.cat([F3, -diff], dim=1))
        w_sum = w1 + w2 + w3 + 1e-8
        fused = (w1 / w_sum) * F1 + (w2 / w_sum) * F2 + (w3 / w_sum) * F3
        return self.conv_adjust(fused)


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8,
                 sampling_rate=0.5, cross_stage=False):
        super().__init__()
        self.initial      = initial
        self.in_channel   = 4 if initial else 6
        self.out_channel  = out_channel
        self.k_num        = k_num
        self.predict      = predict
        self.sr           = sampling_rate
        self.cross_stage  = cross_stage

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True))

        self.pcv = PCV(feat_dim=64, hidden_dim=32, top_k=10)

        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            GNN(int(self.k_num), self.out_channel),
            MSDEA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MSDEA(in_channels=self.out_channel),
            OABlock(self.out_channel, clusters=256),
            MSDEA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))

        self.cgr = CGR(self.out_channel)

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            OABlock(self.out_channel, clusters=128),
            MSDEA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            MSDEA(in_channels=self.out_channel),
            GNN(int(self.k_num), self.out_channel),
            MSDEA(in_channels=self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        if self.cross_stage:
            self.cross_key_fuse = nn.Sequential(
                nn.Conv2d(self.out_channel * 2, self.out_channel, (1, 1)),
                nn.BatchNorm2d(self.out_channel),
                nn.ReLU(inplace=True),
            )

        if self.predict:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if not predict:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2,
                                       index=indices.repeat(1, self.out_channel, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y, cross_key_outs=None):
        B, _, N, _ = x.size()
        x_raw       = x.transpose(1, 3).contiguous()
        coords_raw  = x_raw[:, :4, :, :]

        out = self.conv(x_raw)

        if cross_key_outs is not None and self.cross_stage:
            out = out + self.cross_key_fuse(torch.cat([out, cross_key_outs], dim=1))

        out1 = out

        out = self.embed_0(out)

        consist_prob, sim_feat_128, inject_alpha = self.pcv(out, coords_raw)
        out = out + inject_alpha.abs() * sim_feat_128 * consist_prob

        w0 = self.linear_0(out).view(B, -1)

        out, out2 = self.cgr(out, w0.detach(), x_raw, consist_prob)

        out = self.embed_1(out)

        w1 = self.linear_1(out).view(B, -1)
        out3 = out

        if not self.predict:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)

            indices_feat = indices[:, :int(N * self.sr)].view(B, 1, -1, 1)
            key_outs = []
            for feat in [out1, out2, out3]:
                feat_ds = torch.gather(feat, dim=2,
                                       index=indices_feat.repeat(1, self.out_channel, 1, 1))
                key_outs.append(feat_ds)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], key_outs
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)
            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class VGPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ds_0 = DS_Block(
            initial=True,  predict=False,
            out_channel=128, k_num=9,
            sampling_rate=config.sr,
            cross_stage=False,
        )
        self.ds_1 = DS_Block(
            initial=False, predict=True,
            out_channel=128, k_num=6,
            sampling_rate=config.sr,
            cross_stage=True,
        )
        self.csp = CSP(channels=128, num_keys=3)

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0, key_outs_ds0 = self.ds_0(x, y)

        ds0_all_feat = self.csp(key_outs_ds0)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1, cross_key_outs=ds0_all_feat)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat


MatchMamba = VGPNet
