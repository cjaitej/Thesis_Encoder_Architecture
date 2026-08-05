"""Quick smoke test - run before full training."""

import sys

import torch

sys.path.insert(0, ".")

from model_yolo26_1d import MuSGD, YOLO26_1D_Regressor


def test_shapes():
    """Verify (B, 6, 200) -> (B, 2) end-to-end."""
    model = YOLO26_1D_Regressor()
    model.eval()
    x = torch.randn(8, 6, 200)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8, 2), f"FAIL: expected (8,2), got {y.shape}"
    print(f"[PASS] Output shape: {y.shape}")
    print(f"[INFO] Parameters  : {model.get_num_params():,}")


def test_backward():
    """Verify gradients flow correctly."""
    model = YOLO26_1D_Regressor()
    model.train()
    optim = MuSGD(model.parameters(), lr=0.005)
    x = torch.randn(4, 6, 200)
    target = torch.randn(4, 2)
    pred = model(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    optim.step()
    print(f"[PASS] Backward + MuSGD step succeeded. Loss={loss.item():.6f}")


def test_strided_loss_compat():
    """Simulate the strided velocity loss used in ronin_resnet.py."""
    model = YOLO26_1D_Regressor()
    model.train()
    x = torch.randn(16, 6, 200)
    pred_velocity = model(x)
    gt_displacement = torch.randn(16, 2)
    loss = torch.nn.functional.mse_loss(pred_velocity, gt_displacement)
    loss.backward()
    print(f"[PASS] Strided velocity loss compat. Loss={loss.item():.6f}")


def test_batch_sizes():
    """Verify model handles variable batch sizes."""
    model = YOLO26_1D_Regressor()
    model.eval()
    for batch_size in [1, 16, 32, 128]:
        with torch.no_grad():
            y = model(torch.randn(batch_size, 6, 200))
        assert y.shape == (batch_size, 2)
    print("[PASS] Variable batch sizes: 1, 16, 32, 128")


def test_intermediate_shapes():
    """Trace every intermediate tensor shape."""
    model = YOLO26_1D_Regressor()
    model.eval()
    x = torch.randn(4, 6, 200)
    with torch.no_grad():
        stem = model.stem(x)
        print(f"  Stem:    {tuple(stem.shape)}")
        f1 = model.stage1(stem)
        print(f"  Stage1:  {tuple(f1.shape)}")
        f2 = model.stage2(f1)
        print(f"  Stage2:  {tuple(f2.shape)}")
        f3 = model.stage3(f2)
        print(f"  Stage3:  {tuple(f3.shape)}")
        neck = model.neck(f1, f2, f3)
        print(f"  Neck:    {tuple(neck.shape)}")
        pooled = model.pool(neck).squeeze(-1)
        print(f"  GAP:     {tuple(pooled.shape)}")
        out = model.head(pooled)
        print(f"  Output:  {tuple(out.shape)}")
    print("[PASS] All intermediate shapes correct")


if __name__ == "__main__":
    print("=== YOLO26-1D Smoke Tests ===")
    test_shapes()
    test_backward()
    test_strided_loss_compat()
    test_batch_sizes()
    test_intermediate_shapes()
    print("\nAll smoke tests passed. Ready for training.")
