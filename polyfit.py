import numpy as np
import matplotlib.pyplot as plt

def _polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

def a_example():
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y, copy=False)
    Z = X**2 + Y**2 + np.random.rand(*X.shape)*0.01

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B = Z.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)


def a():
    a = np.arange(0,6)
    b = np.arange(0,6)
    c = a * b
    A,B = np.meshgrid(a, b)
    C = A * B
    print(c,C)


def b():
    a = np.arange(1,6)
    b = np.asarray(a)
    print(b)

def d(x, y, x1, y1):
    z = list(zip(x, y))
    s = np.array(sorted(z))
    s = s.T
    x = s[0]
    y = s[1]

    c, stats = np.polynomial.polynomial.polyfit(x, y, 5, full=True)
    print(c)
    f = np.polynomial.polynomial.polyval(x, c)


    _ = plt.plot(x, y, 'g', label='Original data', markersize=10)
    _ = plt.plot(x, f, 'r', label='Fitted line')

    z = list(zip(x1, y1))
    s = np.array(sorted(z))
    s = s.T
    x = s[0]
    y = s[1]

    c, stats = np.polynomial.polynomial.polyfit(x, y, 5, full=True)
    print(c)
    f = np.polynomial.polynomial.polyval(x, c)

    _ = plt.plot(x, y, 'b', label='Original data', markersize=10)
    _ = plt.plot(x, f, 'y', label='Fitted line')

    plt.show()

def e(x, y):
    z = list(zip(x, y))
    s = np.array(sorted(z))
    s = s.T
    x = s[0]
    y = s[1]

    c, stats = np.polynomial.polynomial.polyfit(x, y, 5, full=True)
    print(c)
    f = np.polynomial.polynomial.polyval(x, c)

    _ = plt.plot(x, y, 'g', label='Original data', markersize=10)
    _ = plt.plot(x, f+100, 'r', label='Fitted line')

    plt.show()

def c():
    left = np.array([101, 54, 274, 209, 214, 211, 214, 213, 216, 210, 198, 92, 100, 83, 83, 99, 95, 91, 50, 52, 44, 51, 50, 45, 54, 44, 91, 115, 88, 89, 487, 235, 256, 255, 251])
    right = np.array([333, 396, 379, 291, 296, 290, 297, 294, 296, 289, 272, 344, 370, 304, 305, 370, 359, 340, 368, 392, 327, 385, 368, 326, 394, 329, 404, 506, 386, 399, 488, 235, 253, 252, 254])
    y = np.array([123, 212, 38, 38, 38, 38, 38, 38, 38, 38, 38, 134, 134, 134, 134, 134, 134, 134, 207, 207, 207, 207, 207, 207, 207, 207, 165, 165, 165, 165, 0, 0, 0, 0, 0])
    x = np.asarray(left) / np.asarray(right)


    # left = np.array([357, 318, 291, 386, 339, 339, 370, 348, 314, 389, 364, 369, 368, 400, 339, 395, 396, 398, 339, 377])
    # right = np.array([208, 201, 182, 243, 208, 211, 109, 104, 93, 115, 107, 111, 109, 61, 59, 63, 61, 63, 51, 58])
    # y1 = np.array([58, 58, 58, 58, 58, 58, 124, 124, 124, 124, 124, 124, 124, 178, 178, 178, 178, 178, 178, 178])
    # x1 = np.asarray(right) / np.asarray(left)
    # d(x, y, x1, y1)

    right = np.append(right,[357, 318, 291, 386, 339, 339, 370, 348, 314, 389, 364, 369, 368, 400, 339, 395, 396, 398, 339, 377])
    left = np.append(left, [208, 201, 182, 243, 208, 211, 109, 104, 93, 115, 107, 111, 109, 61, 59, 63, 61, 63, 51, 58])
    y = np.append(y, [58, 58, 58, 58, 58, 58, 124, 124, 124, 124, 124, 124, 124, 178, 178, 178, 178, 178, 178, 178])
    x = np.asarray(left) / np.asarray(right)
    e(x, y)

    return

    # print(len(left))
    # print(len(right))
    # print(len(y))

    z = list(zip(x, y))
    s = np.array(sorted(z))
    s = s.T
    x = s[0]
    y = s[1]

    # A = np.vstack([x, np.ones(len(x))]).T
    # m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # f = m * x + c

    # x = np.arange(1,10)
    # y = 3*x**2 + 2*x + 8

    c, stats = np.polynomial.polynomial.polyfit(x, y, 5, full=True)
    print(c)
    f = np.polynomial.polynomial.polyval(x, c)


    _ = plt.plot(x, y, 'g', label='Original data', markersize=10)
    _ = plt.plot(x, f, 'r', label='Fitted line')

    _ = plt.legend()

    #
    # fig = plt.figure()
    #
    # fig.add_subplot(231)
    # ax1 = fig.add_subplot(2, 3, 1)  # equivalent but more general
    #
    # fig.add_subplot(232, frameon=False)  # subplot with no frame
    # fig.add_subplot(233, projection='polar')  # polar subplot
    # fig.add_subplot(234, sharex=ax1)  # subplot sharing x-axis with ax1
    # fig.add_subplot(235, facecolor="red")  # red subplot

    #ax1.remove()  # delete ax1 from the figure
    ##fig.add_subplot(ax1)  # add ax1 back to the figure

    plt.show()

#from polynomial2d.polynomial2d import polyfit2d

def main():
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    z = x**2 + y**2

    X, Y = np.meshgrid(x, y, copy=False)
    Z = X**2 + Y**2
    result = _polyfit2d(x, y, Z)
    c = result[1]
    print(c)


    f = np.polynomial.polynomial.polyval2d(x, y, c)
    plt.plot(x, z, color='green')
    plt.plot(x, f, color = 'yellow')

    # print(fitted_surf)
    # a = np.diag(range(15))
    # print(a)
    # a = np.zeros(20,20)
    # plt.matshow(fitted_surf)
    plt.show()

if __name__ == '__main__':
    c()