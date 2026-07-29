"""
Neural residual-refinement stage for RoNIN window-regression backbones.

This is the learned, end-to-end analogue of ResidualNav's Random-Forest corrector
(Toshniwal et al.). The backbone predicts one planar velocity per window; along a
trajectory those predictions form a sequence v_hat[0..N-1]. Small, systematic
per-window errors in that sequence accumulate into trajectory drift (ATE). This
module runs a light bidirectional GRU over the prediction sequence and outputs a
*residual correction*, so the corrected velocity is

    v_tilde = v_hat + r_hat,     r_hat = RefinementNet(v_hat)

which is then integrated into the trajectory. Novelty vs the paper: a learned,
end-to-end recurrent corrector over the prediction sequence instead of a separate
Random Forest over 18 hand-crafted features -- fewer moving parts, and it can be
trained with a drift-aware objective.

Design notes:
  * Input features per step: [vx, vy, speed, dvx, dvy] -- the predicted velocity
    plus its magnitude and first difference (cheap local context for the corrector).
  * The output head is zero-initialised, so at the start r_hat = 0 and v_tilde =
    v_hat exactly (identity). The refinement can only *depart* from the backbone as
    it learns, so it never makes things worse at initialisation.

Input:  [B, N, 2]  (backbone per-window velocity predictions along a trajectory)
Output: [B, N, 2]  (corrected velocities)
"""

import torch
import torch.nn as nn


class RefinementNet(nn.Module):
    def __init__(self, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        in_feat = 5  # vx, vy, speed, dvx, dvy
        self.gru = nn.GRU(
            in_feat, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, 2)
        # identity init: start as v_tilde == v_hat, learn corrections from there
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _features(v):
        """v: [B, N, 2] -> [B, N, 5]"""
        speed = v.norm(dim=-1, keepdim=True)
        dv = torch.zeros_like(v)
        dv[:, 1:] = v[:, 1:] - v[:, :-1]
        return torch.cat([v, speed, dv], dim=-1)

    def forward(self, v):
        """v: [B, N, 2] -> [B, N, 2] (corrected)"""
        out, _ = self.gru(self._features(v))
        return v + self.head(out)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = RefinementNet(hidden=64)
    print(f"RefinementNet params: {m.get_num_params():,}")

    v = torch.randn(2, 500, 2, requires_grad=True)
    out = m(v)
    print(f"forward: {tuple(v.shape)} -> {tuple(out.shape)}")
    assert out.shape == v.shape

    # identity at init: corrected == input (zero-initialised head)
    id_err = (out - v).abs().max().item()
    print(f"identity-at-init max|out - in| = {id_err:.3e}  (should be ~0)")
    assert id_err < 1e-6

    # gradients flow after a step
    out.sum().backward()
    print("grad flows:", all(p.grad is not None for p in m.parameters()))

    # variable-length sequences work (GRU handles any N)
    for n in (37, 1000, 6000):
        assert m(torch.randn(1, n, 2)).shape == (1, n, 2)
    print("variable-length OK (N = 37, 1000, 6000)")
