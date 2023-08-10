from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 12})

# Returns number of samples first-best contract asks to collect
def fbN(p,d,alpha,beta):
    n = (p * d) / (alpha * beta)
    n = pow(n, 1.0/(p+1))
    return n

# Returns the payment offered by the first-best contract
def fbT(p,d,alpha,beta):
    fb_n = fbN(p,d,alpha,beta)
    return alpha * fb_n

def fbUtility(p,d,alpha,beta):
    n_fb = fbN(p,d,alpha,beta)
    t_fb = alpha * n_fb
    u = 0.75 - (d / pow(n,p) - alpha*beta*n)

def optNLow(p, d, alpha, beta,delta,n0=0):
    return(max(n0,fbN(p,d,alpha,beta)))

def optNHigh(p,d,alpha,beta,delta,n0=0):
    def poly(x):
        if(x<=0):
            return(-1)
        a = 2*(pow(d + delta * pow(x,p), (p+1)/p))
        a = 2*alpha*beta*(1 - (pow(d,1/p)/a))
        a = a - (p*d/(pow(x,p+1)))
        return(a)
    n_high = optimize.root(poly, 2).x[0]
    return(max(n0,n_high))

def optTHigh(p,d,alpha,beta,delta, n_high = 0):
    if(n_high == 0):
        n_high = optNHigh(p,d,alpha,beta,delta)
    return(alpha*n_high)

def optTLow(p,d,alpha,beta,delta, n_high = 0, n_low = 0):
    if(n_high == 0):
        n_high = optNHigh(p,d,alpha,beta,delta)
    if(n_low == 0):
        n_low = optNLow(p,d,alpha,beta,delta)
    t = n_high*pow(d,1/p) / (pow(d + delta*pow(n_high, p), 1/p))
    t = alpha * (n_low + n_high - t)
    return(t)

def optPrincipalUtility(p,d,alpha,beta,delta,n0=0):
    n_high = optNHigh(p,d,alpha,beta,delta,n0)
    n_low = optNLow(p,d,alpha,beta,delta,n0)
    t_high = optTHigh(p,d,alpha,beta,delta,n_high)
    t_low = optTLow(p,d,alpha,beta,delta,n_high,n_low)
    alow = 0
    blow = 0
    ahigh = 0
    bhigh = 0
    # print(n_high,n_low,t_high,t_low)
    if(n_low > 0):
        alow = d/pow(n_low,p)
        blow = beta*t_low
    if(n_high > 0):
        ahigh = d/pow(n_high,p)
        bhigh = beta*t_high
    u = 1.5 - alow - ahigh - blow - bhigh
    return(max(0,u/2))

def plotOptUtilityVsDelta(p,d,alpha,beta):
    num_deltas = 50
    delta_lb = 0.0
    delta_ub = 0.22
    diff = (delta_ub - delta_lb) / num_deltas
    utilities = np.zeros(num_deltas)
    deltas = np.zeros(num_deltas)
    for i in range(num_deltas):
        deltas[i] = 2*i*diff 
        utilities[i] = optPrincipalUtility(p,d,alpha,beta,deltas[i])
    plt.plot(deltas, utilities)
    # plt.title("Opt utility vs optimal error gap")
    plt.xlabel(r"$\Delta(\theta)$")
    plt.ylabel("Optimal utility")
    plt.show()

def plotOptUtilityVsP(d,alpha,beta,delta):
    num_ps = 50
    p_lb = 0.5
    p_ub = 3.0
    diff = (p_ub - p_lb) / num_ps
    utilities = np.zeros(num_ps)
    ps = np.zeros(num_ps)
    for i in range(num_ps):
        ps[i] = p_lb + i*diff 
        utilities[i] = max(0,optPrincipalUtility(ps[i],d,alpha,beta,delta))
    plt.plot(ps, utilities)
    # plt.title("Opt utility vs error decay rate")
    plt.xlabel("p")
    plt.ylabel("Optimal utility")
    plt.show()

def plotOptUtilityVsD(p,alpha,beta,delta):
    num_ds = 50
    d_lb = 1
    d_ub = 10
    diff = (d_ub - d_lb) / num_ds
    utilities = np.zeros(num_ds)
    ds = np.zeros(num_ds)
    for i in range(num_ds):
        ds[i] = d_lb + i*diff 
        utilities[i] = max(0,optPrincipalUtility(p,ds[i],alpha,beta,delta))
    plt.plot(ds, utilities)
    # plt.title("Opt utility vs training complexity")
    plt.xlabel("d")
    plt.ylabel("Optimal utility")
    plt.show()

def plotOptUtilityVsAlpha(p,d,beta,delta):
    num_alphas = 50
    alpha_lb = 10
    alpha_ub = 100
    diff = (alpha_ub - alpha_lb) / num_alphas
    utilities = np.zeros(num_alphas)
    alphas = np.zeros(num_alphas)
    for i in range(num_alphas):
        alphas[i] = alpha_lb + i*diff 
        utilities[i] = max(0,optPrincipalUtility(p,d,alphas[i],beta,delta))
        # utilities[i] = 
    plt.plot(alphas, utilities)
    # plt.title("Opt utility vs cost per sample")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Optimal utility")
    plt.show()

def plotOptUtilityVsBeta(p,d,beta,delta):
    num_alphas = 50
    alpha_lb = 0.01
    alpha_ub = 0.05
    diff = (alpha_ub - alpha_lb) / num_alphas
    utilities = np.zeros(num_alphas)
    alphas = np.zeros(num_alphas)
    for i in range(num_alphas):
        alphas[i] = alpha_lb + i*diff 
        utilities[i] = max(0,optPrincipalUtility(p,d,beta,alphas[i],delta))
        # utilities[i] = 
    plt.plot(alphas, utilities)
    # plt.title("Opt utility vs value for accuracy")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Optimal utility")
    plt.show()

def informationRent(p,d,alpha,beta,delta):
    n_high = optNHigh(p,d,alpha,beta,delta)
    n_low = optNLow(p,d,alpha,beta,delta)
    # t_high = optTHigh(p,d,alpha,beta,delta,n_high)
    t_low = optTLow(p,d,alpha,beta,delta,n_high,n_low)
    return(t_low - (alpha*n_low))

def downwardDist(p,d,alpha,beta,delta):
    n_fb = fbN(p,d,alpha,beta)
    n_high = optNHigh(p,d,alpha,beta,delta)
    return(n_fb - n_high)

def plotRentVsGap(p,d,alpha,beta):
    num_deltas = 50
    delta_lb = 0.0
    delta_ub = 0.22
    diff = (delta_ub - delta_lb) / num_deltas
    rents = np.zeros(num_deltas)
    deltas = np.zeros(num_deltas)
    for i in range(num_deltas):
        deltas[i] = 2*i*diff 
        rents[i] = informationRent(p,d,alpha,beta,deltas[i])
    plt.plot(deltas, rents)
    # plt.title("Information rent vs optimal error gap")
    plt.xlabel(r"$\Delta(\theta)$")
    plt.ylabel("Information rent")
    plt.show()

def plotDownwardDistVsGap(p,d,alpha,beta):
    num_deltas = 50
    delta_lb = 0.0
    delta_ub = 0.22
    diff = (delta_ub - delta_lb) / num_deltas
    dists = np.zeros(num_deltas)
    deltas = np.zeros(num_deltas)
    for i in range(num_deltas):
        deltas[i] = 2*i*diff 
        dists[i] = downwardDist(p,d,alpha,beta,deltas[i])
    plt.plot(deltas, dists)
    # plt.title("Downward distortion vs optimal error gap")
    plt.xlabel(r"$\Delta(\theta)$")
    plt.ylabel("Downward distortion magnitude")
    plt.show()

def allowedThetaForFreeAdvSel(p,d,alpha,beta):
    t = alpha*beta*pow(d,1/p)/p 
    t = 1 - pow(p+1,2)*pow(t,p/(p+1))
    return(t)

def plotThetaVsAlphaFreeAdvSelRegion(p,d,beta):
    alpha_ub = p*pow(1/(p+1), 2*(p+1)/p) / (beta*pow(d,1/p))
    alpha_lb = 0
    num_alphas = 100
    diff = (alpha_ub - alpha_lb) / num_alphas
    alphas = np.zeros(num_alphas)
    thetas = np.zeros(num_alphas)
    for i in range(num_alphas):
        alphas[i] = alpha_lb + i*diff 
        thetas[i] = allowedThetaForFreeAdvSel(p,d,alphas[i],beta)
    plt.fill_between(alphas,thetas)
    # plt.title("Problem parameters region where simple contract is good")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\theta$")
    plt.show()


def optPoolingUtility(p,d,alpha,beta,delta):
    def nLow(n):
        y=n*pow(d,1/p) / pow(d + delta*(pow(n,p)), 1/p)
        return(y)

    def poly(x):
        v = pow(d,1/p) / pow(d + delta*(pow(x,p)), (p+1)/p)
        y = nLow(x)
        v = v*p*d / (2*pow(y,p+1))
        v = v + (p*d/(2*pow(x,p+1)))
        return(v-alpha*beta)

    n_high = optimize.root(poly,2).x[0]
    n_low = nLow(n_high)
    t = n_high * alpha
    u = 0.75 - (d/(2*pow(n_low,p))) - (d/(2*pow(n_high,p))) - beta*t
    return(u)

def plotApproxPoolSepVsDelta(p,d,alpha,beta,ks,plot_all=False):
    num_ks = ks.shape[0]
    k_legends = []
    for i in range(num_ks):
        k_legends.append("k="+str(ks[i])+" State learning")
    delta_lb = 0.0
    delta_ub = 0.5
    num_deltas = 100
    diff = (delta_ub - delta_lb) / num_deltas
    pools = np.zeros(num_deltas)
    seps = np.zeros((num_ks,num_deltas))
    opts = np.zeros(num_deltas)
    deltas = np.zeros(num_deltas)
    pools_seps = np.zeros((num_ks,num_deltas))
    for j in range(num_deltas):
        delta = delta_lb + (j+1)*diff
        deltas[j] = delta
        opts[j] = optPrincipalUtility(p,d,alpha,beta,delta)
        print(deltas[j],opts[j])
        pools[j] = optPoolingUtility(p,d,alpha,beta,delta)
    for i in range(num_ks):
        k = ks[i]
        for j in range(num_deltas):
            delta = deltas[j]
            n0 = k/(delta*delta)
            seps[i][j] = optPrincipalUtility(p,d,alpha,beta,delta,n0)
            pools_seps[i][j] = max(seps[i][j],pools[j])
    plt.plot(deltas,opts,label="State aware")
    if plot_all:
        plt.plot(deltas,pools,label="Opt pooling")
        plt.plot(deltas,seps[0],label="Opt separating")
    else:
        for i in range(num_ks):
            plt.plot(deltas,pools_seps[i],label=k_legends[i])
    plt.legend()
    plt.xlabel(r"$\Delta \theta$")
    plt.ylabel("Utility")
    plt.show()

def plotTypeAwareApproxVaryK(p,d,alpha,beta):
    num_ks = 500
    ks = np.zeros(num_ks)
    k_lb_1 = 0.1
    k_ub_1 = 0.17
    diff_1 = (k_ub_1 - k_lb_1)/50
    for i in range(50):
        ks[i] = k_lb_1 + i*diff_1 
    k_lb = k_lb_1 + 49*diff_1
    k_ub = 0.25
    diff = (k_ub - k_lb)/450
    for i in range(450):
        ks[50+i] = k_lb+(i+1)*diff 
    num_deltas = 500
    delta_lb = 0.1
    delta_ub = 0.4
    diff_delta = (delta_ub - delta_lb) / num_deltas
    mult_apprxs = np.zeros(num_ks)
    add_apprxs = np.zeros(num_ks)
    for i in range(num_ks):
        # ks[i] = k_lb + i*diff
        for j in range(num_deltas):
            delta = delta_lb + (j+1)*diff 
            n0 = ks[i] / (delta*delta)
            type_aware = optPrincipalUtility(p,d,alpha,beta,delta)
            sep = optPrincipalUtility(p,d,alpha,beta,delta,n0)
            pool = optPoolingUtility(p,d,alpha,beta,delta)
            type_learn = max(sep,pool)
            add_apprxs[i] = max(add_apprxs[i], type_aware - type_learn)
            mult_apprxs[i] = max(mult_apprxs[i],type_aware / type_learn)
    plt.plot(ks,mult_apprxs)
    plt.xlabel("Testing difficulty parameter (k)")
    plt.ylabel("Utilities ratio state-aware vs state-learning")
    # plt.yticks(fontsize=1)
    plt.show()




p = 0.5
d = 1
alpha = 0.1
beta = 0.1
delta = 0.1
# ks = np.array([0.1,0.2,0.4,0.6,0.8,1.0,2.0])
ks = np.array([1.0,2.0,3.0])

# Figures in the paper

# Figure 1a
plotRentVsGap(p,d,alpha,beta)
# Figure 1b
plotDownwardDistVsGap(p,d,alpha,beta)
# Figure 1c
plotOptUtilityVsDelta(p,d,alpha,beta)

# Figure 2a
plotApproxPoolSepVsDelta(p,d,alpha,beta,np.array([ks[0]]), True)
# Figure 2b
plotApproxPoolSepVsDelta(p,d,alpha,beta,ks)
# Figure 2c
plotTypeAwareApproxVaryK(p,d,alpha,beta)


# plotOptUtilityVsP(d,alpha,beta,delta)
# plotOptUtilityVsAlpha(p,alpha,beta,delta)
# plotOptUtilityVsBeta(p,d,alpha,delta)
# plotOptUtilityVsD(p,alpha,beta,delta)

# plotThetaVsAlphaFreeAdvSelRegion(p,d,beta)







