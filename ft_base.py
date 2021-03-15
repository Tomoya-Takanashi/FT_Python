import numpy as np
from math import exp,e,log,erfc,inf,erf

def logarithm(value):
    if value==0:
        return -inf
    try:
        return log(value)
    except ValueError:
        return -inf

def exparithm(value):
    try:
        return exp(value)
    except OverflowError:
        if value>0:
            return inf
        else:
            return 0

def Cal_FL(orbital_E,fermi,theta):
    FL = []
    
    for i in range(len(orbital_E)):
        if theta == 0:
            FL.append(0)
        else:
            temp = 1+exparithm((orbital_E[i]-fermi)/theta)
            FL.append(1/temp)

    return FL

def Cal_FT_Ene(fermi,Orbital_E,fock,mo,theta,hcore):

    val = 0
    FL = Cal_FL(Orbital_E,fermi,theta)
    size = len(FL)
    #print(fermi)
    new_D = np.zeros((size,size))
    for ao1 in range(size):
        for ao2 in range(size):
            for a in range(size):
                val = val + np.complex(FL[a]).real * mo[ao1][a] * np.conjugate(mo)[ao2][a]
            new_D[ao1][ao2] = val
            val = 0
            
    Ene = np.trace(np.dot(new_D,fock+hcore))/2
    entropy = 0

    for i in range(len(FL)):
        if FL[i] == 0:
            term1 = 0
        else:
            term1 = np.complex(FL[i]).real*logarithm(np.complex(FL[i]).real)
            
        if FL[i] == 1:
            term2 = 0
        else:
            term2 = (1-np.complex(FL[i]).real) * logarithm(1-np.complex(FL[i]).real)
            
        entropy = entropy + theta*(term1 +term2)

    return Ene + entropy


def E_cal(fon,orbital_E,theta):
    E_val = 0
    for i in range(len(fon)):
        if fon[i] == 0:
            term1 = 0
        else:
            term1 = np.complex(fon[i]).real*logarithm(np.complex(fon[i]).real)
            
        if fon[i] == 1:
            term2 = 0
        else:
            term2 = (1-np.complex(fon[i]).real) * logarithm(1-np.complex(fon[i]).real)

        E_val = E_val + theta*(term1 + term2) + np.complex(fon[i]).real*orbital_E[i]
    return E_val

def I_cal(fon,orbital_E,theta):
    I_val = 0
    for i in range(len(fon)):
        if fon[i] == 0:
            term1 = 0
        else:
            term1 = np.complex(fon[i]).real*logarithm(np.complex(fon[i]).real)
            
        if fon[i] == 1:
            term2 = 0
        else:
            term2 = (1-np.complex(fon[i]).real) * logarithm(1-np.complex(fon[i]).real)
    
        I_val = I_val + theta*(term1+term2) + np.complex(fon[i]).real * orbital_E[i]
    return I_val
        
def A_cal(fon,orbital_E,theta):
    A_val = 0
    for i in range(len(fon)):
        if fon[i] == 0:
            term1 = 0
        else:
            term1 = np.complex(fon[i]).real*logarithm(np.complex(fon[i]).real)
            
        if fon[i] == 1:
            term2 = 0
        else:
            term2 = (1-np.complex(fon[i]).real) * logarithm(1-np.complex(fon[i]).real)
            
        A_val = A_val + theta*(term1 + term2) + np.complex(fon[i]).real*orbital_E[i]

    return A_val
