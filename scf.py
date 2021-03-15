import numpy as np
from scipy import optimize
from scipy.linalg import eig,eigh
from math import exp,e,log,erfc,inf,erf
from pyscf import gto,scf

def make_nes(mol):

    alpha = int(mol.nelec[0])
    beta = int(mol.nelec[1])
    ene_nuc = gto.mole.energy_nuc(mol)
    mol_scf = scf.hf.SCF(mol)
    mol = mol_scf.mol
    ao2e = mol.intor('int2e')
    S = mol.intor('int1e_ovlp')
    size = len(S)
    s,U = np.linalg.eigh(S)
    U_trans = np.conjugate(U.T)
    s = s ** (-1/2)
    s_zero = np.zeros((size,size))
    for i in range(size):
        s_zero[i][i] = s[i]
    X = np.dot(U,s_zero)
    hcore = mol_scf.get_hcore(mol)
    
    d_init = scf.uhf.get_init_guess(mol) 
    d_alpha_init = d_init[0]
    d_beta_init = d_init[1]
    return alpha,beta,hcore,X,ao2e,ene_nuc,S,d_alpha_init,d_beta_init

def scf_roothan(alpha,beta,hcore,X,d_alpha,d_beta,size,ao2e,DEBUG=False):

    size = len(hcore)
    new_D_alpha = np.zeros((size,size))
    new_D_beta = np.zeros((size,size))
    X_trans = np.conjugate(X.T)
    fock_alpha = np.zeros((size,size))
    fock_beta = np.zeros((size,size))
    val = 0
    d_Total = d_alpha + d_beta

    for ao1 in range(size):
        for ao2 in range(size):
            for ao3 in range(size):
                for ao4 in range(size):
                    val = val + ( d_Total[ao3][ao4]*ao2e[ao1][ao2][ao4][ao3] - d_alpha[ao3][ao4]*ao2e[ao1][ao3][ao4][ao2] )
            fock_alpha[ao1][ao2] = val
            val = 0
    val = 0
    fock_alpha = fock_alpha+hcore

    for ao1 in range(size):
        for ao2 in range(size):
            for ao3 in range(size):
                for ao4 in range(size):
                    val = val + ( d_Total[ao3][ao4]*ao2e[ao1][ao2][ao4][ao3] - d_beta[ao3][ao4]*ao2e[ao1][ao3][ao4][ao2] )
            fock_beta[ao1][ao2] = val
            val = 0
    val = 0
    fock_beta = fock_beta+hcore

    new_fock_alpha = np.dot(np.dot(X_trans,fock_alpha),X)
    new_fock_beta = np.dot(np.dot(X_trans,fock_beta),X)

    e_alpha,C_alpha = eigh(new_fock_alpha)
    e_beta,C_beta = eigh(new_fock_beta)
    new_C_alpha = np.dot(X,C_alpha)
    new_C_beta = np.dot(X,C_beta)
    lumo_alpha = e_alpha[int(alpha-1)]
    homo_alpha = e_alpha[int(alpha)]
    f_alpha = (homo_alpha + lumo_alpha)/2
    lumo_beta = e_beta[int(beta-1)]
    homo_beta = e_beta[int(beta)]
    f_beta = (homo_beta + lumo_beta)/2
    f_init = [f_alpha,f_beta]

    for ao1 in range(size):
        for ao2 in range(size):
            for a in range(alpha):
                val = val + new_C_alpha[ao1][a] * np.conjugate(new_C_alpha)[ao2][a]
            new_D_alpha[ao1][ao2] = val
            val = 0
    val = 0

    for ao1 in range(size):
        for ao2 in range(size):
            for a in range(beta):
                val = val + new_C_beta[ao1][a] * np.conjugate(new_C_beta)[ao2][a]
            new_D_beta[ao1][ao2] = val
            val = 0

    return new_D_alpha,new_D_beta,fock_alpha,fock_beta,new_C_alpha,new_C_alpha,e_alpha,e_beta

