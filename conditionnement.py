import numpy as np
import matplotlib.pyplot as plt


def matrix_conditionnement(A, b, Delta_b):
    """
    Solve linear problem b = Ax for varying measurement b with incertitude Delta_b.
    incertitude on measure b so that b' = b + Delta_b.
    Conditionnement on matrix A is calculated for the l2 norm as
                    kappa(A) = sigma_max(A)/sigma_min(A)
                    where sigma_max and sigma_min are the maximum and minimum singular values of A.
    Relative error on b is given by ||Delta_b||/||b||
    Idem for the relative error on x.
    :param A:
    :type A: matric (NxN)
    :param b:
    :type b: array(Nx1)
    :param Delta_b:
    :type Delta_b: array(Nx1)
    :return: Relative error on the matrix, b and x.
    :rtype:
    """
    b_prime = b + Delta_b
    RE_b = np.linalg.norm(Delta_b)/np.linalg.norm(b)

    # x' = A^(-1)*b
    x = np.dot(np.linalg.inv(A), b)
    x_prime = np.dot(np.linalg.inv(A), b_prime)

    Delta_x = x_prime - x
    RE_x = np.linalg.norm(Delta_x)/np.linalg.norm(x)

    # Calculate the SVD of A = USV^H, take the matrix S
    sigma_A = np.linalg.svd(A)[1]
    # Singular value are ordrerd from max to min so that sigma_max = sigma[0] and sigma_min = sigma[-1]
    kappa_A = sigma_A[0]/sigma_A[-1]
    return kappa_A, RE_b, RE_x


if __name__ == "__main__":
    A = np.array([[7, 1, 11, 10], [2, 6, 5, 2], [8, 11, 3, 8], [6, 9, 3, 6]])
    b = np.array([29, 15, 30, 24])
    Delta_b = np.array([1, 1, 1, 1])

    kappa_A, Re_b, RE_x = matrix_conditionnement(A, b, Delta_b)
    print(f' kappa(A) = {kappa_A:.0f};\n RelativeError on b = {Re_b:.4f};\n Relative error on x = {RE_x:.4f}')