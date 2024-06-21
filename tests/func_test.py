import ex2 as hw
import numpy as np
def test_hwfunc1():
    model, R2 = hw.homework()
    assert np.allclose(model.coef_[0,0], 3.52576347, atol=1e-4)
    assert np.allclose(model.intercept_, 29.33199771, atol=1e-4)


def test_hwfunc2():
    model, R2 = hw.homework()
    assert np.allclose(R2, 0.984182776801406, atol=1e-4)
  
