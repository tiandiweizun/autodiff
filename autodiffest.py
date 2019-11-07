from autodiff import Tensor
import numpy as np


def test_identity():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)
    y = x2
    assert np.array_equal(y.data, x2_val)
    y.backward()
    assert np.array_equal(x2.grad, np.ones_like(x2_val))


def test_add_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)
    y = 5 + x2
    assert np.array_equal(y.data, x2_val + 5)
    y.backward()
    assert np.array_equal(x2.grad, np.ones_like(x2_val))


def test_sub_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)
    y = np.ones(3) - x2
    assert np.array_equal(y.data, np.ones(3) - x2_val)
    y.backward()
    assert np.array_equal(x2.grad, -np.ones_like(x2_val))


def test_sub_two_vars():
    x1_val = 2 * np.ones(3)
    x2_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    y = x1 - x2

    assert np.array_equal(y.data, x1_val - x2_val)
    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))
    assert np.array_equal(x2.grad, - np.ones_like(x2_val))


def test_neg():
    x2_val = 2 * np.ones(3)
    x1_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    y = -x2 + x1
    assert np.array_equal(y.data, -x2_val + x1_val)
    y.backward()
    assert np.array_equal(x2.grad, -np.ones_like(x2_val))
    assert np.array_equal(x1.grad, np.ones_like(x1_val))


def test_mul_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)
    y = 5 * x2
    assert np.array_equal(y.data, x2_val * 5)
    y.backward()
    assert np.array_equal(x2.grad, np.ones_like(x2_val) * 5)


def test_div_two_vars():
    x1_val = 2 * np.ones(3)
    x2_val = 5 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    y = x1 / x2
    assert np.array_equal(y.data, x1_val / x2_val)
    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val) / x2_val)
    assert np.array_equal(x2.grad, -x1_val / (x2_val * x2_val))


def test_div_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)
    y = 5 / x2
    assert np.array_equal(y.data, 5 / x2_val)
    y.backward()
    assert np.array_equal(x2.grad, -5 / (x2_val * x2_val))


def test_add_two_vars():
    x1_val = 2 * np.ones(3)
    x2_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    y = x1 + x2
    assert np.array_equal(y.data, x1_val + x2_val)
    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))
    assert np.array_equal(x2.grad, np.ones_like(x2_val))


def test_mul_two_vars():
    x1_val = 2 * np.ones(3)
    x2_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    y = x1 * x2
    assert np.array_equal(y.data, x1_val * x2_val)
    y.backward()
    assert np.array_equal(x1.grad, x2_val)
    assert np.array_equal(x2.grad, x1_val)


def test_add_mul_mix_1():
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    x3 = Tensor(x3_val)
    y = x1 + x2 * x3 * x1
    assert np.array_equal(y.data, x1_val + x2_val * x3_val)
    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val) + x2_val * x3_val)
    assert np.array_equal(x2.grad, x3_val * x1_val)
    assert np.array_equal(x3.grad, x2_val * x1_val)


def test_add_mul_mix_2():
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    x3 = Tensor(x3_val)
    x4 = Tensor(x4_val)
    y = x1 + x2 * x3 * x4
    assert np.array_equal(y.data, x1_val + x2_val * x3_val * x4_val)
    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))
    assert np.array_equal(x2.grad, x3_val * x4_val)
    assert np.array_equal(x3.grad, x2_val * x4_val)
    assert np.array_equal(x4.grad, x2_val * x3_val)


def test_add_mul_mix_3():
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x2 = Tensor(x2_val)
    x3 = Tensor(x3_val)
    z = x2 * x2 + x2 + x3 + 3
    y = z * z + x3

    z_val = x2_val * x2_val + x2_val + x3_val + 3
    expected_yval = z_val * z_val + x3_val
    expected_grad_x2_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
    expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1

    assert np.array_equal(y.data, expected_yval)
    y.backward()
    assert np.array_equal(x2.grad, expected_grad_x2_val)
    assert np.array_equal(x3.grad, expected_grad_x3_val)


def test_matmul_two_vars():
    x2_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3
    x2 = Tensor(x2_val)
    x3 = Tensor(x3_val)
    y = x2.matmul(x3)
    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))
    assert np.array_equal(y.data, expected_yval)
    y.backward()
    assert np.array_equal(x2.grad, expected_grad_x2_val)
    assert np.array_equal(x3.grad, expected_grad_x3_val)


def test_log_op():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    y = x1.log()
    assert np.array_equal(y.data, np.log(x1_val))
    y.backward()
    assert np.array_equal(x1.grad, 1 / x1_val)


def test_log_two_vars():
    x1_val = 2 * np.ones(3)
    x2_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    y = (x1 * x2).log()

    assert np.array_equal(y.data, np.log(x1_val * x2_val))
    y.backward()
    assert np.array_equal(x1.grad, x2_val / (x1_val * x2_val))
    assert np.array_equal(x2.grad, x1_val / (x1_val * x2_val))


def test_exp_op():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    y = x1.exp()
    assert np.array_equal(y.data, np.exp(x1_val))
    y.backward()
    assert np.array_equal(x1.grad, np.exp(x1_val))


def test_exp_mix_op():
    x1_val = 2 * np.ones(3)
    x2_val = 4 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)
    y = ((x1 * x2).log() + 1).exp()
    assert np.array_equal(y.data, np.exp(np.log(x1_val * x2_val) + 1))
    y.backward()
    assert np.array_equal(x1.grad, y.data * x2_val / (x1_val * x2_val))
    assert np.array_equal(x2.grad, y.data * x1_val / (x1_val * x2_val))


def test_reduce_sum():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    y = x1.sum()
    assert np.array_equal(y.data, np.sum(x1_val))
    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))


def test_reduce_sum_mix():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    y = x1.sum().exp()
    expected_y_val = np.exp(np.sum(x1_val))
    assert np.array_equal(y.data, expected_y_val)
    y.backward()
    assert np.array_equal(x1.grad, expected_y_val * np.ones_like(x1_val))
    x1.zero_gard()
    y2 = x1.sum().log()
    expected_y2_val = np.log(np.sum(x1_val))
    assert np.array_equal(y2.data, expected_y2_val)
    y2.backward()
    assert np.array_equal(x1.grad, (1 / np.sum(x1_val)) * np.ones_like(x1_val))


def test_mix_all():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    y = 1 / (1 + (-x1.sum()).exp())
    expected_y_val = 1 / (1 + np.exp(-np.sum(x1_val)))
    expected_y_grad = expected_y_val * (1 - expected_y_val) * np.ones_like(x1_val)
    assert np.array_equal(y.data, expected_y_val)
    y.backward()
    assert np.sum(np.abs(x1.grad - expected_y_grad)) < 1E-10


def test_logistic():
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.zeros(3)
    x1 = Tensor(x1_val)
    w = Tensor(w_val)
    y = 1 / (1 + (-(w * x1).sum()).exp())
    expected_y_val = 1 / (1 + np.exp(-np.sum(w_val * x1_val)))
    expected_y_grad = expected_y_val * (1 - expected_y_val) * x1_val
    assert np.array_equal(y.data, expected_y_val)
    y.backward()
    assert np.sum(np.abs(w.grad - expected_y_grad)) < 1E-7


def test_log_logistic():
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.zeros(3)
    x1 = Tensor(x1_val)
    w = Tensor(w_val)
    y = (1 / (1 + (-(w * x1).sum()).exp())).log()
    logistic = 1 / (1 + np.exp(-np.sum(w_val * x1_val)))
    expected_y_val = np.log(logistic)
    expected_y_grad = (1 - logistic) * x1_val
    assert np.array_equal(y.data, expected_y_val)
    y.backward()
    assert np.sum(np.abs(w.grad - expected_y_grad)) < 1E-7


def test_logistic_loss():
    y_val = 0
    x_val = np.array([2, 3, 4])
    w_val = np.random.random(3)
    x = Tensor(x_val)
    w = Tensor(w_val)
    y = Tensor(y_val)
    h = 1 / (1 + (-(w * x).sum()).exp())
    l = y * h.log() + (1 - y) * (1 - h).log()
    logistic = 1 / (1 + np.exp(-np.sum(w_val * x_val)))
    expected_L_val = y_val * np.log(logistic) + (1 - y_val) * np.log(1 - logistic)
    expected_w_grad = (y_val - logistic) * x_val
    assert expected_L_val == l.data
    l.backward()
    assert np.sum(np.abs(expected_w_grad - w.grad)) < 1E-9


def test_mean():
    x1_val = np.array([1, 2, 3, 4])
    x = Tensor(x1_val)
    y = x.mean()
    assert np.array_equal(y.data, np.mean(x1_val))
    y.backward()
    assert np.array_equal(x.grad, np.ones(x1_val.shape) / 4)
