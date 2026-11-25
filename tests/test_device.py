import pytest
import torch


def test_qutip_imports():
    import qutip

    assert int(qutip.__version__.split(".")[0]) >= 5
    assert qutip.jmat(1, "z").shape == (3, 3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matmul():
    a = torch.randn(512, 512, device="cuda")
    b = a @ a.T
    torch.cuda.synchronize()
    assert torch.allclose(b.cpu(), a.cpu() @ a.cpu().T, rtol=1e-3, atol=1e-2)
