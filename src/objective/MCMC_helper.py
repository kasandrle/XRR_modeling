import numpy as np
import matplotlib.pyplot as plt

def gaussian_init(x0, lb, ub, nwalkers, scale=0.1):
    """
    x0: array of shape (ndim,)
    lb, ub: arrays of shape (ndim,)
    scale: fraction of parameter range used as std dev
    """
    x0 = np.asarray(x0)
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    ndim = len(x0)

    # Standard deviation per dimension
    sigma = (ub - lb) * scale

    # Draw Gaussian samples
    walkers = x0 + np.random.randn(nwalkers, ndim) * sigma

    # Enforce bounds
    walkers = np.clip(walkers, lb, ub)

    return walkers



def plottrace_bf(sampler,steps,cols,best_fit=None):
    
    plt.style.use('bmh')
    sh = np.shape(sampler)
    numPloz = int((len(cols)) / 2)+1
    f, axarr = plt.subplots(2,numPloz, sharex=True,figsize=(30,15))
    chain = sampler
    
    try:
        for k in range(numPloz):
            for i in range(sh[0]):
                axarr[0,k].plot(chain[i,0:steps,k], color='k', alpha=0.1)
                axarr[0,k].axhline(best_fit[k])
                axarr[0,k].set_ylabel(cols[k])
        for k in range(numPloz,len(cols)):
            for i in range(sh[0]):
                axarr[1,k-numPloz].plot(chain[i,0:steps,k], color='k', alpha=0.1)
                axarr[1,k-numPloz].axhline(best_fit[k])
                axarr[1,k-numPloz].set_ylabel(cols[k])
        for i in range(numPloz):
            axarr[1,i].set_xlabel('Steps')
        
    except:
        for k in range(numPloz):
            for i in range(sh[0]):
                axarr[0,k].plot(chain[i,0:steps,k], color='k', alpha=0.1)
                #axarr[0,k].axhline(best_fit[k])
                axarr[0,k].set_ylabel(cols[k])
        for k in range(numPloz,len(cols)):
            for i in range(sh[0]):
                axarr[1,k-numPloz].plot(chain[i,0:steps,k], color='k', alpha=0.1)
                #axarr[1,k-numPloz].axhline(best_fit[k])
                axarr[1,k-numPloz].set_ylabel(cols[k])
        for i in range(numPloz):
            axarr[1,i].set_xlabel('Steps')
    f.subplots_adjust(hspace=0)
    #f.savefig('chain_pce.png')
    #plt.close(f)
    return 0.0

def plottrace(sampler,steps,cols):
    
    plt.style.use('bmh')
    sh = np.shape(sampler)
    numPloz = int((len(cols)) / 3)+1
    f, axarr = plt.subplots(3,numPloz, sharex=True,figsize=(30,15))
    chain = sampler
    
    for k in range(numPloz):
        for i in range(sh[1]):
            axarr[0,k].plot(chain[0:steps,i,k], color='k', alpha=0.1)
                #axarr[0,k].axhline(best_fit[k])
            axarr[0,k].set_ylabel(cols[k])
    for k in range(numPloz,2*numPloz):
        for i in range(sh[1]):
            axarr[1,k-numPloz].plot(chain[0:steps,i,k], color='k', alpha=0.1)
                #axarr[1,k-numPloz].axhline(best_fit[k])
            axarr[1,k-numPloz].set_ylabel(cols[k])
    for k in range(2*numPloz,len(cols)):
        for i in range(sh[1]):
            axarr[2,k-2*numPloz].plot(chain[0:steps,i,k], color='k', alpha=0.1)
                #axarr[1,k-numPloz].axhline(best_fit[k])
            axarr[2,k-2*numPloz].set_ylabel(cols[k])
    
    for i in range(numPloz):
        axarr[1,i].set_xlabel('Steps')
    f.subplots_adjust(hspace=0)
    #f.savefig('chain_pce.png')
    #plt.close(f)
    return 0.0
