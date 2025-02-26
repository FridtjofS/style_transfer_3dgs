import torch
import torchvision


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

        z_best_batch = torch.argmin(d_mat, 2)
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)

    return z_best


def nn_feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()


class NNFMLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = torchvision.models.vgg16(pretrained=True).eval().to(device)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def forward(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nnfm_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
    ):
        for x in loss_names:
            assert x in ["nnfm_loss", "content_loss", "gram_loss"]

        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.get_feats(styles, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        loss_dict = dict([(x, 0.0) for x in loss_names])
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            if "nnfm_loss" in loss_names:
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict["nnfm_loss"] += cos_loss(x_feats, target_feats)

        return loss_dict
