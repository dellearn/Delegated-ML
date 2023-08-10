from scipy import optimize
from scipy.stats import norm
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
import math


def opt_utility(alpha, theta=0.0, d=1.0, beta=1.0, p=0.5, base=1.0):
    u = (
            1.0
            - theta
            - ((p + 1) * math.pow(alpha * beta * math.pow(d, 1.0 / p) / p, p / (p + 1)))
    )
    return u


def opt_contract_linear(alpha, theta=0.0, d=1.0, beta=1.0, p=0.5, base=1.0):
    # Constraint to ensure IR
    th_ir = alpha * math.pow(d, 1.0 / p) / p
    th_ir = th_ir * math.pow((p + 1) / (base - theta), (p + 1) / p)
    c = 1.0 / (beta * math.pow(p + 1, (p + 1) / p))
    if c < th_ir:
        c = th_ir

    return min(1, c)


def utility_linear_contract(c, alpha, theta=0.0, d=1.0, beta=1.0, p=0.5, base=1.0):
    n = pow(p*c*d/alpha, 1 / (p+1))
    acc_n = 1 - theta - (d / pow(n, p))
    agent_utility = (c * acc_n) - (alpha * n)
    if agent_utility < 0:
        return 0
    utility = (1 - beta * c) * acc_n
    return max(0, utility)


def opt_utility_linear(alpha, theta=0.0, d=1.0, beta=1.0, p=0.5, base=1.0):
    c = opt_contract_linear(alpha, theta, d, beta, p, base)
    return utility_linear_contract(c, alpha, theta, d, beta, p, base)


def alphaMaxThetaAgnostic(theta, d, beta=1.0, p=0.5, base=1.0):
    t1 = 4 * (1 - theta) * (1 - theta) * (1 - theta) / (27 * d * d)
    t = p * pow((base - theta) / pow((p + 1), 2), (p + 1) / p) / (beta * math.pow(d, 1.0 / p))
    return t


def alphaMaxContracting(theta, d, beta=1.0, p=0.5, base=1.0):
    t = p * pow((base - theta) / (p+1), (p + 1) / p) / (beta * math.pow(d, 1.0 / p))
    return t

def thetaMaxThetaAgnostic(alpha, d, beta=1.0, p=0.5, base=1.0):
    t = base - pow(p+1, 2) * pow(alpha * beta * pow(d, 1/p) / p, p / (p + 1))
    return t
#
#
def thetaMaxContracting(alpha, d, beta=1.0, p=0.5, base=1.0):
    t = base - (p + 1) * pow(alpha * beta * pow(d, 1/p) / p, p / (p + 1))
    return t


def theta_agnostic_contract_higher_cost(d, beta=1.0, p=0.5, base=1.0):
    num_alphas = 10
    thetas = [0.2, 0.6, 0.8]
    utilities = np.zeros((len(thetas), num_alphas))
    c = opt_contract_linear(0, 0, d, beta, p, base)
    alpha_bounds = [0,0]
    for j in range(len(thetas)):
        theta = thetas[j]
        lb_alpha = 0
        ub_alpha = alphaMaxContracting(theta, d, beta, p, base)
        diff = (ub_alpha - lb_alpha) / (num_alphas + 1)
        alphas = np.zeros(num_alphas)
        for i in range(num_alphas):
            alpha = lb_alpha + (i+1) * diff
            alphas[i] = alpha
            opt = opt_utility(alpha, theta, d, beta, p, base)
            lin_utility = utility_linear_contract(c, alpha, theta, d, beta, p, base)
            utilities[j][i] = lin_utility / opt

        plt.plot(alphas, utilities[j], label='theta='+str(theta))

    plt.legend()
    plt.xlabel('Alpha')
    plt.ylabel('Approximation ratio of Theta-agnostic linear contract')
    plt.title('Approximation ratio of theta-agnostic contract for different costs')
    plt.show()

def theta_agnostic_contract_different_thetas(d, beta=1.0, p=0.5, base=1.0):
    num_thetas = 10
    alphas = [0.1, 0.2]
    utilities = np.zeros((len(alphas), num_thetas))
    c = opt_contract_linear(0, 0, d, beta, p, base)
    for j in range(len(alphas)):
        alpha = alphas[j]
        lb_theta = 0
        t = thetaMaxThetaAgnostic(alpha,d,beta,p,base)
        ub_theta = thetaMaxContracting(alpha, d, beta, p, base)
        diff = (ub_theta - lb_theta) / (num_thetas + 1)
        thetas = np.zeros(num_thetas)
        for i in range(num_thetas):
            theta = lb_theta + (i+1) * diff
            thetas[i] = theta
            opt = opt_utility(alpha, theta, d, beta, p, base)
            lin_utility = utility_linear_contract(c, alpha, theta, d, beta, p, base)
            utilities[j][i] = lin_utility / opt

        plt.plot(thetas, utilities[j], label='alpha='+str(alpha))

    plt.legend()
    plt.xlabel('Theta')
    plt.ylabel('Approximation ratio of Theta-agnostic linear contract')
    plt.title('Approximation ratio of theta-agnostic contract for different theta values')
    plt.show()


theta_agnostic_contract_higher_cost(1, 0.1, 1)
theta_agnostic_contract_different_thetas(1, 0.1, 1)

