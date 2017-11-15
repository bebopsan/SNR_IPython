# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:29:26 2017

Some investigations on SNR behaviour using values from standards as much as 
possible.

Based on: 
    
    "Noise in Fiber optic communication links" 
    by:  Robert Dahlgren 

@author: santiagoe
"""

import numpy as np
from scipy.special import erfc,erfcinv
from scipy.constants import k, elementary_charge
from math import log10
import matplotlib.pyplot as plt
from sympy import symbols,Eq, solveset
x, y, z = symbols('x y z')

plt.close('all')

def mw_to_dbm(mW):
    """This function converts a power given in mW to a power given in dBm."""
    return 10.*log10(mW)

def dbm_to_w(dBm):
    """This function converts a power given in dBm to a power given in mW."""
    return 10**((dBm)/10.)*1e-3


B   = 10e9   # Bandwidth Hz
T   = 300    # 25 degrees C
R   = 1e3    # PD resistance
RIN = -145   # RIN noise from laser (dB/Hz) from spec
Q_target = 7.05 # Target Q factor for BER 1e-12


i_D = 10e-9  # Dark current spec (Amp)  Dahlgren example
#i_D = 6e-4  # Mellanox estimate spec (Amp)  
i_T = 2*np.sqrt(k*T*B/R)  # Thermal current noise
i_tx_rms = 1e-6 # Laser driver RMS noise current  (Amp)


TP3_OMA = -4.4 # Receiver OMA outer (dBm)

P_r = 4.2    # Maximum received power (dBm) 

eta_LD = 0.5 # Laser diode slope efficiency in W/A
eta_pd = 0.9 # PD thermal noise resistance  (Ohm)
eta_rx = 1e3 # PD responsivity              (A/W)
eta_fo = 1   # Laser 
 
sigma_Tx = 5e-7    # Tx circuit Watt RMS
sigma_Tx = eta_LD*eta_fo*i_tx_rms   # Tx circuit Watt RMS

sigma_D  = 6.3e-9 # Dark current noise Watt RMS
sigma_D  = (1/eta_pd)*np.sqrt(2*elementary_charge* i_D*B) # Dark current noise Watt RMS

sigma_shot = (1/eta_pd)*np.sqrt(2*elementary_charge*eta_pd * B*  dbm_to_w(TP3_OMA)) # Shot noise Watt RMS 
sigma_shot_factor = (1/eta_pd)*np.sqrt(2*elementary_charge*eta_pd *B) 

sigma_rin  = eta_fo*np.sqrt(RIN*B)*dbm_to_w(TP3_OMA) # Laser RIN noise distribution Watt RMS
sigma_rin_factor  = eta_fo*np.sqrt(10**(0.1*RIN)*B)

sigma_john = (1/eta_pd)*(1/eta_rx)*2*np.sqrt(k*T*B*R)


sigma_1_sq = sigma_D**2 + sigma_Tx**2 + sigma_john**2

sigma_1 = np.sqrt(sigma_1_sq)
sigma_1_eq = sigma_D**2 + sigma_Tx**2 + sigma_john**2 \
                +sigma_shot_factor**2*x+sigma_rin_factor**2*x**2

sigma_0 = np.sqrt(2.07e-13)
sigma_0 = np.sqrt(sigma_john**2 + sigma_D**2)

noise_eq = Eq((x/Q_target**2-sigma_0)**2,sigma_1_eq)
a = solveset(noise_eq,x)
I_target = a.args[1] # Power for Q target in W
I_target_dBm = mw_to_dbm(I_target*1e3) # Power for Q target in dBm 

Q = dbm_to_w(TP3_OMA)/(sigma_john) 
#Q = dbm_to_w(TP3_OMA)/(sigma_1) 


SNR_dBm = 10*np.log10(Q**2)


OMA =  np.linspace(-35,0,16)
Q_vec = dbm_to_w(OMA)/(sigma_john) 
#Q_vec = dbm_to_w(OMA)/(sigma_1) 
SNR_dBm_vec = 10*np.log10(Q_vec)

#------------ Plots -----------------------------

l = plt.semilogy(OMA,Q_vec,'-')
#plt.legend((l1, l2), ('l1','l2'))
plt.xlabel('OMA (dBm)')
plt.ylabel('Q ')
plt.title('NRZ: T=25 C B = 10 GHz R = 1e3 Ohm')
plt.grid(True, which ="both")
plt.show()

l = plt.plot(OMA,SNR_dBm_vec,'-')
#plt.legend((l1, l2), ('l1','l2'))
plt.xlabel('OMA (dBm)')
plt.ylabel('SNR (dB) ')
plt.title('')
plt.grid(True, which ="both")
plt.show()


# Noise variance vs optical power

sigma_1_sq = sigma_D**2 + sigma_Tx**2 + sigma_john**2 + \
             sigma_shot_factor**2*dbm_to_w(OMA) + sigma_rin_factor**2*dbm_to_w(OMA)**2
sigma_1 = np.sqrt(sigma_1_sq)
l = plt.semilogy(OMA,sigma_1 + sigma_0 ,'-')
#plt.legend((l1, l2), ('l1','l2'))
plt.xlabel('OMA (dBm)')
plt.ylabel('$\sigma_{1}+\sigma_{0}$ (Watt RMS)')
plt.title('')
plt.grid(True, which ="both")
plt.show()

# Q Factor vs optical power with all sources of variance

Q_vec = np.sqrt(dbm_to_w(OMA)/(sigma_1+sigma_0)) 
#Q_vec = dbm_to_w(OMA)/(sigma_1) 
SNR_dBm_vec = 10*np.log10(Q_vec**2)

plt.figure()
l = plt.plot(OMA,Q_vec,'-')
#plt.legend((l1, l2), ('l1','l2'))
plt.axvline(x=[I_target_dBm], color='k', linestyle='--')
plt.xlabel('OMA (dBm)')
plt.ylabel('$Q$ ')
plt.title('')
plt.grid(True, which ="both")
plt.show()

plt.figure()
l = plt.plot(OMA,SNR_dBm_vec,'-')
#plt.legend((l1, l2), ('l1','l2'))
plt.axvline(x=[I_target_dBm], color='k', linestyle='--')
plt.xlabel('OMA (dBm)')
plt.ylabel('$SNR$(dB) ')
plt.title('')
plt.grid(True, which ="both")
plt.show()

BER_Q = lambda Q: 1/2*erfc(Q/np.sqrt(2))
plt.figure()
l = plt.semilogy(Q_vec,BER_Q(Q_vec),'-')
plt.axvline(x=[Q_target], color='k', linestyle='--')
#plt.legend((l1, l2), ('l1','l2'))
plt.xlabel('Q')
plt.ylabel('$BER$ ')
plt.title('')
plt.grid(True, which ="both")
plt.show()
#                    Normal Distribution


Q_BER = lambda BER: np.sqrt(2)*erfcinv(2*BER)


import numpy as np
import matplotlib.pyplot as plt 

def make_gauss(N, sig, mu):
    return lambda x: N/(sig * (2*np.pi)**.5) * np.e ** (-(x-mu)**2/(2 * sig**2))

def main():
    ax = plt.figure().add_subplot(1,1,1)
    x = np.arange(-5, 5, 0.01)
    s = np.sqrt([0.2, 1, 5, 0.5])
    m = [0, 0, 0, -2] 
    c = ['b','r','y','g']

    for sig, mu, color in zip(s, m, c): 
        gauss = make_gauss(1, sig, mu)(x)
        ax.plot(x, gauss, color, linewidth=2)

    plt.xlim(-5, 5)
    plt.ylim(0, 1)
    plt.legend(['0.2', '1.0', '5.0', '0.5'], loc='best')
    plt.show()

if __name__ == '__main__':
   main()