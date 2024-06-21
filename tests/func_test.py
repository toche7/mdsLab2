import ex2 as hw
import numpy as np
def test_hwfunc1():
    model, R2 = hw.homework()
    assert np.allclose(model.coef_[0,0], 2.3514381737209162, atol=1e-4)
    assert np.allclose(model.intercept_, 21.27552289, atol=1e-4)


def test_hwfunc2():
    model, R2 = hw.homework()
    assert np.allclose(R2, 0.984182776801406, atol=1e-4)
  
