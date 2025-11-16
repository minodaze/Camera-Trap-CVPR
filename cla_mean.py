class FeatureHead(nn.Module):
    """Return raw (pre-softmax) features from the base classifier."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        if hasattr(self.base, "forward_features"):
            print("Using forward_features to extract features.")
            feats = self.base.forward_features(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            feats = feats.float().view(feats.size(0), -1)
            return feats
        else:
            logits = self.base(x)
            return logits.float()

@torch.no_grad()
def class_mean_distance_ref_vs_target(feat_model: nn.Module, ref_loader, target_loader) -> float:
    """
    L2 twice:
      (1) L2-normalize each per-class mean vector for REF and TARGET
      (2) Per-class distance = L2( mean_ref_norm - mean_tgt_norm ), then average
    """
    feat_model.eval()

    def class_means(loader):
        sums, counts = defaultdict(lambda: None), defaultdict(int)
        for batch in loader:
            if isinstance(batch, dict):
                x = batch.get("images") or batch.get("pixel_values") or batch.get("input")
                y = batch.get("labels") or batch.get("label")
            else:
                x, y = batch[0], batch[1]
            x = x.to(DEVICE, non_blocking=True)
            feats = feat_model(x).detach().cpu()
            for f, lab in zip(feats, y):
                lab = int(lab)
                if sums[lab] is None:
                    sums[lab] = f.clone()
                else:
                    sums[lab] += f
                counts[lab] += 1

        means = {}
        for k, c in counts.items():
            mu = sums[k] / c
            mu = mu / (mu.norm(p=2) + 1e-12)  # first L2-normalization
            means[k] = mu
        return means

    ref_mu    = class_means(ref_loader)
    tgt_mu    = class_means(target_loader)

    common = sorted(set(ref_mu.keys()) & set(tgt_mu.keys()))
    if not common:
        return float("nan")

    dists = []
    for k in common:
        d = (ref_mu[k] - tgt_mu[k]).norm(p=2).item()  # second L2 (distance between normalized means)
        dists.append(d)

    return float(np.mean(dists))





