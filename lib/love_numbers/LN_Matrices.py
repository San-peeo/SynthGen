# MATRICES
from lib.lib_dep import *

# Credit to Dr. Consorzi Anastasia's work



#Elastic Direct Matrix
def Y(radius, rho, mu, g, l):

    M=mp.zeros(6,6, dtype=mp.mpc)

    M[0,0]=(l*radius**(l+1))/(2*(2*l+3))
    M[0,1]=radius**(l-1)
    M[0,3]=((l+1)*radius**(-l))/(2*(2*l-1))
    M[0,4]=radius**(-l-2)

    M[1,0]=((l+3)*radius**(l+1))/(2*(2*l+3)*(l+1))
    M[1,1]=(radius**(l-1))/l
    M[1,3]=((2-l)*radius**(-l))/(2*l*(2*l-1))
    M[1,4]=(-radius**(-l-2))/(l+1)

    M[2,0]=((l*rho*g*radius**(l+1))/(2*(2*l+3)))+(((l**2-l-3)*mu*radius**l)/((2*l+3)))
    M[2,1]=((rho*g)/(radius**(-l+1)))+((2*(l-1)*mu)/(radius**(-l+2)))
    M[2,2]=rho*radius**l
    M[2,3]=((l+1)*radius*rho*g /(2*(2*l-1)*radius**(l+1))) + ((-(l**2)-3*l+1)*mu/((2*l-1)*radius**(l+1)))
    M[2,4]=(rho*g*(radius**(-l-2)))-((2*(l+2)*mu)/(radius**(l+3)))
    M[2,5]=rho/(radius**(l+1))

    M[3,0]=(l*(l+2)*mu*radius**l)/(((2*l+3)*(l+1)))
    M[3,1]=(2*(l-1)*mu*radius**(l-2))/l
    M[3,3]=(mu*(l**2-1))/(l*(2*l-1)*radius**(l+1))
    M[3,4]=(2*(l+2)*mu)/((l+1)*radius**(l+3))

    M[4,2]=radius**l
    M[4,5]=1/(radius**(l+1))

    M[5,0]=(2*np.pi*G_const*rho*l*radius**(l+1))/(2*l+3)
    M[5,1]=4*np.pi*G_const*rho*radius**(l-1)
    M[5,2]=(2*l+1)/radius**(-l+1)
    M[5,3]=(2*np.pi*G_const*rho*(l+1))/((2*l-1)*radius**l)
    M[5,4]=(4*np.pi*G_const*rho)/(radius**(l+2))

    return M

# --------------------------------------------------------------------------------------------------------

#Elastic Inverse Matrix
def Y_inv(radius, rho, mu, g, l):

    N=mp.zeros(6,6, dtype=mp.mpc)
    D=mp.zeros(6,6, dtype=mp.mpc)


    N[0,0]=(rho*g*radius/mu) -(2*(l+2))
    N[0,1]=2*l*(l+2)
    N[0,2]=-radius/mu
    N[0,3]=(l*radius)/mu
    N[0,4]=(rho*radius)/mu

    N[1,0]=(-(rho*g*radius/mu))+(2*(l**2+3*l-1)/(l+1))
    N[1,1]=-2*(l**2-1)
    N[1,2]=radius/mu
    N[1,3]=((2-l)*radius)/mu
    N[1,4]=-(rho*radius)/mu

    N[2,0]=4*np.pi*G_const*rho
    N[2,5]=-1

    N[3,0]=(rho*g*radius/mu)+(2*(l-1))
    N[3,1]=2*((l**2)-1)
    N[3,2]=-radius/mu
    N[3,3]=(-(l+1)*radius)/mu
    N[3,4]=(rho*radius)/mu

    N[4,0]=(-(rho*g*radius/mu))-(2*(l**2-l-3)/l)
    N[4,1]=-2*l*(l+2)
    N[4,2]=radius/mu
    N[4,3]=(l+3)*radius/mu
    N[4,4]=-rho*radius/mu

    N[5,0]=4*np.pi*G_const*rho*radius
    N[5,4]=2*l+1
    N[5,5]=-radius

    D[0,0]=((l+1)*radius**(-l-1))/(2*l+1)
    D[1,1]=l*(l+1)*radius**(-l+1)/(2*(2*l+1)*(2*l-1))
    #D[2,2]=-radius**(-l+1)/(2*l+1) (- =TYPO???)
    D[2,2]=-radius**(-l+1)/(2*l+1)
    D[3,3]= (l*radius**l)/(2*l+1)
    D[4,4]=(l*(l+1)*radius**(l+2))/(2*(2*l+1)*(2*l+3))
    D[5,5]=radius**(l+1)/(2*l+1)

    return D*N

# --------------------------------------------------------------------------------------------------------

#Inviscid core Matrix
def Icf( radius, rho, mu, g, l):

    M=mp.zeros(6,3, dtype=mp.mpc)


    M[0,0]=-radius**l/g
    M[0,2]=1

    M[1,1]=1
    
    M[2,2]=g*rho
    
    M[4,0]=radius**l
    
    M[5,0]=2*(l-1)*radius**(l-1)
    M[5,2]=4*np.pi*G_const*rho

    return M






# --------------------------------------------------------------------------------------------------------



#Elastic core
def Ic(radius, rho, mu, g, l):

    M=mp.zeros(6,3, dtype=mp.mpc)


    M[0,0]=(l*radius**(l+1))/(2*(2*l+3))
    M[0,1]=radius**(l-1)

    M[1,0]=((l+3)*radius**(l+1))/(2*(2*l+3)*(l+1))
    M[1,1]=(radius**(l-1))/l
    

    M[2,0]=((l*rho*g*radius**(l+1))/(2*(2*l+3)))+(((l**2-l-3)*mu*radius**l)/((2*l+3)))
    M[2,1]=((rho*g)/(radius**(-l+1)))+((2*(l-1)*mu)/(radius**(-l+2)))
    M[2,2]=rho*radius**l
   

    M[3,0]=(l*(l+2)*mu*radius**l)/(((2*l+3)*(l+1)))
    M[3,1]=(2*(l-1)*mu*radius**(l-2))/l
   
    M[4,2]=radius**l
    

    M[5,0]=(2*np.pi*G_const*rho*l*radius**(l+1))/(2*l+3)
    M[5,1]=4*np.pi*G_const*rho*radius**(l-1)
    M[5,2]=(2*l+1)/radius**(-l+1)

    return M




# --------------------------------------------------------------------------------------------------------


def Hf(radius, rho, g, l):
    
    # mu = complex(mu[0],mu[1])


    M=mp.zeros(2,2, dtype=mp.mpc)



    M[0,0]=radius**l
    M[0,1]=radius**(-l-1)

    #M[1,0]=((2*l+1)*radius**(l-1))-(4*np.pi*G_const*rho/g)*radius**l
    #M[1,1]=-(4*np.pi*G_const*rho/g)radius**(-l-1)

    
    M[1,0]=l*radius**(l-1)
    M[1,1]=-(l+1)*radius**(-l-2)

    return M





   
