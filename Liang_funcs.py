import numpy as np
import math


def IF_np(x1, x2, dx1, dx2, dt=1):
    """
    calculate Information Flow by Liang and normalized one by Bai.
    """
    
    N = np.count_nonzero(~np.isnan(x1 * x2))

    C = np.cov([x1, x2])
    #print(C)

    dC = np.zeros_like(C)
    dC[0, 0] = np.nansum((x1 - np.nanmean(x1)) * (dx1 - np.nanmean(dx1)))
    dC[0, 1] = np.nansum((x1 - np.nanmean(x1)) * (dx2 - np.nanmean(dx2)))
    dC[1, 0] = np.nansum((x2 - np.nanmean(x2)) * (dx1 - np.nanmean(dx1)))
    dC[1, 1] = np.nansum((x2 - np.nanmean(x2)) * (dx2 - np.nanmean(dx2)))
    dC = dC / (N - 1)
    #print("dC ", dC)

    detC = np.linalg.det(C)
    #print("detC ", detC)

    a11 = C[1, 1] * dC[0, 0] - C[0, 1] * dC[1, 0]
    a12 = -C[0, 1] * dC[0, 0] + C[0, 0] * dC[1, 0]

    a11 = a11 / detC
    a12 = a12 / detC
    #print("a11 ", a11)
    #print("a12 ", a12)

    f1 = np.nanmean(dx1) - a11 * np.nanmean(x1) - a12 * np.nanmean(x2)
    #print("f1 ", f1)

    R1 = dx1 - (f1 + a11 * x1 + a12 * x2)
    #print("R1 ", R1)

    Q1 = np.nansum(R1 * R1)
    #print("Q1 ", Q1)

    b1 = np.sqrt(Q1 * dt / N)
    #print("b1 ", b1)


    # covariance matrix
    NI = np.zeros((4,4))
    NI[0,0] = N * dt / b1/b1
    NI[1,1] = dt/b1/b1 * np.nansum(x1 * x1)
    NI[2,2] = dt/b1/b1 * np.nansum(x2 * x2)
    NI[3,3] = 3*dt/(b1**4) * np.nansum(R1 * R1) - N/b1/b1
    NI[0, 1] = dt/b1/b1 * np.nansum(x1)
    NI[0, 2] = dt/b1/b1 * np.nansum(x2)
    NI[0, 3] = 2*dt/(b1**3) * np.nansum(R1)
    NI[1, 2] = dt*b1/b1 * np.nansum(x1 * x2)
    NI[1, 3] = 2*dt/(b1**3) * np.nansum(R1 * x1)
    NI[2, 3] = 2*dt/(b1**3) * np.nansum(R1 * x2)

    NI[1, 0] = NI[0, 1]
    NI[2, 0] = NI[0, 2]
    NI[2, 1] = NI[1, 2]
    NI[3, 0] = NI[0, 3]
    NI[3, 1] = NI[1, 3]
    NI[3, 2] = NI[2, 3]
    #####
    invNI = np.linalg.inv(NI)
    var_a12 = invNI[2,2]

    T21 = C[0, 1] / C[0, 0] * (-C[1, 0] * dC[0, 0] + C[0, 0] * dC[1, 0]) / detC
    #T21 = np.abs(T21)
    var_T21 = (C[0,1] / C[0, 0])**2 * var_a12

    dH1_star = a11
    dH1_noise = b1 ** 2 / (2.0 * C[0, 0])
    #print("dH1_noise", dH1_noise)

    Z = np.abs(T21) + np.abs(dH1_star) + np.abs(dH1_noise)

    # normalize
    #tau21b = np.abs(T21) / (np.abs(dH1_noise) + np.abs(T21))
    #tau21l = np.abs(T21) / Z
    tau21b = T21 / (np.abs(dH1_noise) + np.abs(T21))
    tau21l = T21 / Z

    # significance level
    z99, z95, z90 = [2.58, 1.96, 1.65] 
    #err99, err95, err90 = np.sqrt(np.abs(var_T21)) * np.array([z99, z95, z90])
    err99, err95, err90 = np.sqrt(var_T21) * np.array([z99, z95, z90])

    # return
    return np.array([T21, tau21l, tau21b, err99, err95, err90])
