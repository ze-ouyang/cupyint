import sys
import os
import cupy as cp
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))# to parent dir  
from cupyint import trapz_integrate, simpson_integrate, booles_integrate, gauss_integrate, mc_integrate, adpmc_integrate
from cupyint import set_backend, get_data_type

data_type = cp.float32

@pytest.fixture(autouse=True)
def setup():
    print("setup")
    data_type = cp.float32
    set_backend(data_type)

@pytest.fixture
def simple_1d_func():
    def f(x):
        return x**2
    return f

@pytest.fixture
def simple_2d_func():
    def f(x, y):
        return x**2 + y**2
    return f

@pytest.mark.parametrize("integrate", [
    trapz_integrate,
    simpson_integrate,
    booles_integrate,
    gauss_integrate
])
def test_1d_polynomial(integrate, simple_1d_func):
    # Integrate x^2 from 0 to 1, should be 1/3
    bounds = [[0, 1]]
    num_points = [101]  # Use odd number for Simpson's rule
    result = integrate(simple_1d_func, None, bounds, num_points, None)
    expected = 1/3
    assert abs(float(cp.asnumpy(result).item()) - expected) < 1e-4

@pytest.mark.parametrize("integrate", [
    trapz_integrate,
    simpson_integrate,
    booles_integrate,
    gauss_integrate
])
def test_2d_polynomial(integrate, simple_2d_func):
    bounds = [[0, 1], [0, 1]]
    num_points = [101, 101]
    result = integrate(simple_2d_func, None, bounds, num_points, None)
    expected = 2/3
    assert abs(float(cp.asnumpy(result).item()) - expected) < 1e-4

def test_3d_with_params():
    def function(x, y, z, params):
        a = params[0]
        b = params[1]
        c = params[2]
        return a * cp.exp(-b * (x**2 + y**2 + z**2)) + c * cp.sin(x) * cp.cos(y) * cp.exp(z)

    def boundary(x, y, z):
        return (x**2 + y**2 + z**2 < 10).astype(cp.float64)

    a_values = cp.linspace(1.0, 10.0, 10000, dtype=data_type)
    b_values = cp.linspace(2.0, 20.0, 10000, dtype=data_type)
    c_values = cp.linspace(0.5, 5, 10000, dtype=data_type)
    param_values = cp.stack((a_values, b_values, c_values), axis=1)

    bounds = [[0, 1], [0, 1], [0, 1]]
    num_points = [33, 33, 33]

    # Test with different integration methods
    methods = [trapz_integrate, simpson_integrate, booles_integrate, gauss_integrate]
    results = []

    for method in methods:
        result = method(function, param_values, bounds, num_points, boundary)
        results.append(result)

    # Check that all methods give similar results
    for i in range(len(results)-1):
        diff = cp.abs(results[i] - results[i+1])
        assert cp.all(diff < 1e-3), f"Methods {i} and {i+1} differ by {cp.max(diff).item()}"
