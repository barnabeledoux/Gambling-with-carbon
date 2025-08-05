import numpy as np
import matplotlib.pyplot as plt
import random as rand
import scipy
from scipy.optimize import fsolve
import os
from colorsarr import colorsarr #For the colormap
colarr = colorsarr()

class kelly_debt:
    """
    Initialize the class with different parameters:
    - n: number of horses
    - L: Leverage
    - Ci: Initial capital of the gambler
    - tp: time of payback
    - rho: interest rate
    - Tmax: maximum time for the simulation
    - iskelly: boolean to indicate if Kelly criterion is used
    - pini: initial probabilities of winning for each horse
    - rini: initial returns for each horse
    - rhoini: initial interest rate
    - gaussian: boolean to indicate if Gaussian approximation for every jump is used
    - W: mean of the Gaussian distribution for the jumps (average of log of gamma)
    - sig: standard deviation of the distribution of log of gamma
    - borroweachstep: boolean to indicate if the gambler borrows at each step
    - D0: initial debt
    """
    def __init__(self, n, L, Ci, tp, rho, Tmax, iskelly = True, pini=[], rini=[], rhoini = 0., gaussian = False, W = 0., sig=0.1, borroweachstep = False, D0=0.):
        self.n, self.B0, self.tp, self.rho, self.Tmax, self.iskelly, self.L, self.Ci = n, (L-1)*Ci, tp, rho, Tmax, iskelly, L, Ci
        self.D0 = D0
        self.rx = self.random_discrete_distribution(self.n)
        self.px = self.random_discrete_distribution(self.n)
        self.C = self.L*self.Ci
        self.D = self.B0
        self.borroweachstep = borroweachstep
        self.tau = 0
        self.Clist = [self.L*self.Ci]
        if len(pini)>0: #If an initial distribution probability is specified
            self.px = pini
        if len(rini)>0: #If an initial distribution of returns is specified
            self.rx = rini
        if self.iskelly: #If Kelly criterion is used, bx = px
            self.bx = self.px
        else:
            self.bx = self.random_discrete_distribution(self.n)
        self.ox = np.array([1/r for r in self.rx])
        self.mean = sum([self.px[i]*np.log(self.bx[i]*self.ox[i]) for i in range(self.n)])
        if rhoini != 0.:
            self.rho = rhoini
        self.gaussian = gaussian
        if gaussian : #If Gaussian approximation is used, W and sig are specified, and p, r, b are not used anymore
            self.W = W
            self.rho = rho
            self.sig = sig
            self.carr = colarr.carr

    def random_discrete_distribution(self, n):
        # Generate n random numbers
        random_numbers = np.random.rand(n)
        # Normalize the numbers so they sum to 1
        distribution = random_numbers / np.sum(random_numbers)
        return distribution

    def emissions(self, tf, intensity, tau0, predic = False): #Compute the emissions of the gambler between tau0 and tf, given the carbon intensity of the game
        gamma, I0 = -(intensity[-1] - intensity[0])/intensity.size, intensity[0] #Get linear parameters to model carbon intensity
        E, Ctot = 0., 0.
        self.initial()
        self.tau = tau0
        while self.tau<tf: #While the time is less than the final time, evolve stochastically
            if self.gaussian:
                self.step_gaussian(self.tau)
            else:
                self.step(self.tau)
            self.tau += 1
            Ctot += self.C
            E += intensity[int(self.tau - tau0)]*self.C if not(predic) else self.C*(I0 - gamma*(self.tau - tau0)) #Update cumulative emissions
        return E, Ctot
    
    def step_gaussian(self, tau):
        if self.C>=0: #If the capital is positive, we can gamble
            jump = np.exp(np.random.normal(self.W, self.sig))
            Cstart = self.C+0.
            self.C = jump*Cstart
            self.D = self.rho*self.D
            if self.borroweachstep: #If the gambler borrows at each step, update the capital and debt
                self.D += (self.L - 1.)*Cstart
                self.C += (self.L - 1.)*Cstart
            if tau == self.tp: #If the time is equal to the payback time, the gambler pays back the debt
                self.C -= self.D
                self.D = 0
        self.Clist.append(self.C) #Update the list of capital
        self.Dlist.append(self.D) #Update the list of debt

    def survprob(self, tfin = 50, repet=1E5): #Compute the array of survival probabilities of the gambler for all times lower than tfin from simulations
        repet = int(repet)
        count, countot = np.zeros(int(tfin)), np.zeros(int(tfin)) #Initialize the count arrays
        for i in range(repet):
            self.initial()
            while self.tau<tfin:
                countot[self.tau] += 1 #Update the total count
                if self.gaussian:
                    self.step_gaussian(self.tau)
                else:
                    self.step(self.tau)
                if self.C > self.D:
                    count[self.tau] += 1 #Update the count of gamblers with capital greater than debt to account for survival
                self.tau += 1
        return count/countot

    def initial(self): #Initialize the parameters for the simulation
        self.tau = 0
        self.C = self.L*self.Ci
        self.D = (self.L - 1.)*self.Ci + self.D0
        self.Clist = [self.L*self.Ci]
        self.Dlist = [(self.L - 1.)*self.Ci]

    def step(self, tau): #Step of the simulation if the gaussian approximation is not used
        p = np.random.rand()
        i=0
        if self.C>=0:
            i = np.random.choice(range(self.n), p=self.px)
            self.C = self.bx[i]*self.ox[i]*self.C
            self.D = self.rho*self.D
            if tau == self.tp:
                self.C -= self.D
                self.D = 0
        self.Clist.append(self.C)
        self.Dlist.append(self.D)
    
    def evol(self): #Evolve the gambler's capital and debt over time
        self.initial()
        while self.tau<self.Tmax:
            if self.gaussian: #If Gaussian approximation is used
                self.step_gaussian(self.tau)
            else:
                self.step(self.tau)
            self.tau += 1

    def plot_evol(self, nsimu, scale='log'): #Plot the stochastic evolution of the gambler's capital over time for nsimu simulations
        fig1 = plt.figure() 
        ax1 = fig1.add_subplot(111)
        ax1.set_yscale(scale)
        ax1.set_xlabel(r'Time $\tau$', fontsize=15)
        ax1.set_ylabel(r'Evolution of the capital $C$', fontsize=15)
        self.count = 0
        self.counttot = 0
        rangetrend = range(10,20)
        trend = [1E4*np.exp((self.mean + (self.L-1 if self.borroweachstep else 0.))*i) for i in rangetrend]
        ax1.plot(rangetrend, trend, ls=':', lw=2., c=self.carr[-1])
        if scale=='log':
            ax1.text(0, 1E4*np.exp((self.mean + (self.L-1 if self.borroweachstep else 0.))*21), r'$\propto\exp(W \tau)$', fontsize = 15)
        else:
            ax1.text(0, 1E-10*np.exp((self.mean + (self.L-1 if self.borroweachstep else 0.))*21), r'$\propto\exp(W \tau)$', fontsize = 15)
        for j in range(nsimu):
            self.evol()
            self.count += int(self.C > 1)
            self.counttot += 1
            k = np.random.randint(1,13)
            ax1.plot(range(self.Tmax+1), self.Clist, c = self.carr[k], alpha = 0.35, lw=2.2)
        xbank = 1-self.count/self.counttot
        if scale == 'log':
            ax1.set_ylim(1E-1, 1E1*np.exp((self.mean + (self.L-1 if self.borroweachstep else 0.))*self.Tmax))
        else:
            ax1.set_ylim(1E-1, 1E-10*np.exp((self.mean + (self.L-1 if self.borroweachstep else 0.))*self.Tmax))
        ax1.set_title(r'$ W=$'+r'$ {}$'.format(str((self.mean + (self.L-1 if self.borroweachstep else 0.)))[:5]) + r', $\,\log(\rho) = $' + r' ${}$'.format(str(np.log(self.rho))[:5]) + r', $\, x_{\text{bankrupt}} = $' + r'${}$'.format(str(xbank)[:5]), fontsize=16)
        cd = os.getcwd()
        fig1.savefig(cd + r'/results/simu_kelly'+str(self.L)+'.pdf')