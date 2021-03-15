import sys
import numpy as np
from pyscf import gto,scf
from scf import make_nes,scf_roothan
from ft_base import Cal_FT_Ene,A_cal,E_cal,I_cal,Cal_FL
from scipy import optimize
from scipy.linalg import eig,eigh
from scipy.special import erfc
from math import exp,e,log,inf,erf

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

def SCF(mol,theta_init,theta_temp,delta_zero):

    alpha,beta,hcore,X,ao2e,ene_nuc,S,d_alpha,d_beta = make_nes(mol)
    size = len(hcore)
    e_new = 0
    error = 1
    ite = 0

    while abs(error) > 1e-10:
        e_old = e_new

        #-----SCF CAL-----#
        result = scf_roothan(alpha,beta,hcore,X,d_alpha,d_beta,size,ao2e,DEBUG=0)

        """
        Alpha_Density = result[0]
        Beta_Density = result[1]
        Alpha_Fock = result[2]
        Beta_Fock = result[3]
        Alpha_MO = result[4]
        Beta_MO = result[5]
        Alpha_OE = result[6]
        Beta_OE =result[7]
        
        """

        ######################################
        #------------FT section -------------#
        ######################################

        homo_alpha = result[6][int(alpha-1)]
        lumo_alpha = result[6][int(alpha)]
        f_alpha = (homo_alpha + lumo_alpha)/2
        
        homo_beta = result[7][int(beta-1)]
        lumo_beta = result[7][int(beta)]
        f_beta = (homo_beta + lumo_beta)/2
        
        f_init = [f_alpha,f_beta]

        FOCK = [result[2],result[3]]
        MO = [result[4],result[5]]
        OE = [result[6],result[7]]
        f_lev = [f_alpha,f_beta]
        
        bound_alpha =[homo_alpha,lumo_alpha]
        bound_beta = [homo_beta,lumo_beta]

        #print(bound_alpha)
        bounds = [bound_alpha,bound_beta]
        error_theta = 1
        ite_ft = 0
        FT_ENERGY_NEW = 0
        theta = theta_init

        
        while abs(error_theta) > 1e-8:
            #FT_ENERGY_OLD = FT_ENERGY_NEW

            #######################################
            #-----Fermi Level Iteration Start-----#
            #######################################
            
            #------- Lagulange Function Define -------#
            def L_min_alpha(fermi):
                return Cal_FT_Ene(fermi,OE[0],FOCK[0],MO[0],theta,hcore)

            def L_min_beta(fermi):
                return Cal_FT_Ene(fermi,OE[1],FOCK[1],MO[1],theta,hcore)
            #-----------------------------------------#

            #------ Constraint Function Define -----#
            
            def cons_alpha(fermi):
                return sum(Cal_FL(result[6],fermi,theta))-alpha
            def cons_beta(fermi):
                return sum(Cal_FL(result[7],fermi,theta))-beta

            cons_A = (
                {'type':'eq','fun':cons_alpha}
            )
            
            cons_B=(
                {'type':'eq','fun':cons_beta}
            )

            #----------------------------------------#
            
            #-----Fermi Level Decision-----#
            res_alpha = optimize.minimize(L_min_alpha,x0=f_alpha,constraints=cons_A,method='trust-constr')
            res_beta = optimize.minimize(L_min_beta,x0=f_init[1],constraints=cons_B,method='trust-constr')
            #------------------------------#

            #-----Occupation number calculation-----#
            fon_alpha = Cal_FL(result[6],res_alpha.x,theta)
            fon_beta = Cal_FL(result[7],res_beta.x,theta)
            #---------------------------------------#

#            print(fon_alpha)
            ########################
            #-----theta update-----#
            ########################

            #------ Constraint Function Define ------#
            def cons_alpha_I(fermi):
                return sum(Cal_FL(result[6],fermi,theta))-int((alpha-1))
            def cons_alpha_EA(fermi):
                return sum(Cal_FL(result[6],fermi,theta))-int((alpha+1))

            def cons_beta_I(fermi):
                return sum(Cal_FL(result[7],fermi,theta))-int((beta-1))
                
            def cons_beta_EA(fermi):
                return sum(Cal_FL(result[7],fermi,theta))-int((beta+1))

            cons_A_I = (
                {'type':'eq','fun':cons_alpha_I}
            )
            cons_A_EA = (
                {'type':'eq','fun':cons_alpha_EA}
            )
            cons_B_I=(
                {'type':'eq','fun':cons_beta_I}
            )
            cons_B_EA=(
                {'type':'eq','fun':cons_beta_EA}
            )
            # ------------------------------------- #


            #------------ Function Define -----------------#
            def minimize_alpha(fermi):
                return Cal_FT_Ene(fermi,OE[0],FOCK[0],MO[0],theta,hcore)

            def minimize_beta(fermi):
                return Cal_FT_Ene(fermi,OE[1],FOCK[1],MO[1],theta,hcore)
            # ------------------------------------------------------#
            
            #-----Fermi level decision-----#
            res_alpha_I = optimize.minimize(minimize_alpha,x0=f_init[0],constraints=cons_A_I,method='trust-constr')
            res_alpha_A = optimize.minimize(minimize_alpha,x0=f_init[0],constraints=cons_A_EA,method='trust-constr')
            res_beta_I = optimize.minimize(minimize_beta,x0=f_init[1],constraints=cons_B_I,method='trust-constr')
            res_beta_A = optimize.minimize(minimize_beta,x0=f_init[1],constraints=cons_B_EA,method='trust-constr')
            #------------------------------#
            #print(OE)

            #-----Ocupation number Caluclation-----#
            fon_alpha_I = Cal_FL(result[6],res_alpha_I.x,theta)
            fon_alpha_A = Cal_FL(result[6],res_alpha_A.x,theta)
            fon_beta_I = Cal_FL(result[7],res_beta_I.x,theta)
            fon_beta_A = Cal_FL(result[7],res_beta_A.x,theta)
            #--------------------------------------#

            #-----IP & EA Calculation-----#
            I_alpha = I_cal(fon_alpha_I,result[6],theta) - E_cal(fon_alpha,result[6],theta)
            A_alpha = - A_cal(fon_alpha_A,result[6],theta) + E_cal(fon_alpha,result[6],theta)
            I_beta = I_cal(fon_beta_I,result[7],theta) - E_cal(fon_beta,result[7],theta)
            A_beta = - A_cal(fon_beta_A,result[7],theta) + E_cal(fon_beta,result[7],theta)

            if I_alpha > I_beta:
                I = I_beta
            else:
                I = I_alpha
        
            if A_alpha > A_beta:
                A = A_alpha
            else:
                A = A_beta
            #----------------------------#
        
            theta_old = theta

            #-----new theta calcation-----#
            delta_FT = I-A
            
            theta = theta_temp * erfc(delta_FT / delta_zero)
            ite_ft = ite_ft + 1
            #-----------------------------#
            #print(ite_ft)
            #print(res_alpha_I.x)
            error_theta_old = error_theta
            error_theta = abs(theta-theta_old)
            #print(ite_ft,theta)
            #print(theta,res_alpha_I)
            if ite_ft > 200:
                #print('FTSCF not converge')
                break
            if error_theta_old == error_theta:
                break
            #####################################
            #-----Fermi level Iteration End-----#
            #####################################

        #-----Entropy Energy Calculation-----#
        def energy_cal(FL,theta_new):
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
                
                entropy = entropy + np.complex(theta_new).real*(term1.real +term2.real)
            return entropy

        entropy_e = energy_cal(fon_alpha,theta) + energy_cal(fon_beta,theta)
        #-------------------------------------#

        #-----New density matrix construction-----#
        new_D_alpha = np.zeros((size,size))
        new_D_beta = np.zeros((size,size))
        val = 0

        for ao1 in range(size):
            for ao2 in range(size):
                for a in range(size):
                    val = val + np.complex(fon_alpha[a]).real * MO[0][ao1][a] * np.conjugate(MO[0])[ao2][a]
                new_D_alpha[ao1][ao2] = val
                val = 0
        val = 0

        for ao1 in range(size):
            for ao2 in range(size):
                for a in range(size):
                    val = val + np.complex(fon_beta[a]).real * MO[1][ao1][a] * np.conjugate(MO[1])[ao2][a]
                new_D_beta[ao1][ao2] = val
                val = 0
        #----------------------------------------#
    
        #-----New total Energy Caculation-----#
        ite = ite + 1
        d_alpha = new_D_alpha
        d_beta = new_D_beta
        densityAlpha_fockAlpha = np.dot(new_D_alpha,result[2]+hcore)
        densityBeta_fockBeta = np.dot(new_D_beta,result[3]+hcore)
        e = np.trace(densityAlpha_fockAlpha)+np.trace(densityBeta_fockBeta)
        e = e * 0.5
        e_new = e + entropy_e + ene_nuc
        #-------------------------------------#
        error = abs(e_new - e_old)
        if ite > 200:
            print('SCF not converge')
            break

    return e_new,fon_alpha,fon_beta,MO[0],MO[1]
