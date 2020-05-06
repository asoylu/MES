# MES: A Python Code for Calculations of Half-lives of Alpha (AD), 
# Cluster Decay (CD) and Spontaneous Fission (SF), 
# Prediction of Decay Modes of Superheavy Nuclei written by A. Soylu 
# at October 2019 in Nigde/TURKIYE

import numpy as np
from  matplotlib import pyplot as plt
import scipy
from scipy import integrate
from scipy.optimize import fsolve
pi=scipy.pi

#Enter Mass,proton number of the parent nuclei
A=286.0
z=113.0
#Enter Mass, proton number of the cluster
A_2=4.0
z_2=2.0
#Enter Q-value of decay
q=8.965
#Enter the shell correction energy for KPS 
Es=7.50


#Mass, proton number of the core
A_1=A-A_2
z_1=z-z_2
n=A-z

#Function for calculating KPS Model for SF half-lives
def kps(A_1,z_1,A_2,z_2,q,Es):
    n_1=A_1-z_1
    n_2=A_2-z_2
    n_t=(n_1+n_2)
    z_t=(z_1+z_2)
    A_t=A_1+A_2 
    a=-43.25203
    b=0.49192
    c=3674.3927
    d=-9360.6
    e=0.8930
    f=578.56058
    kps1=a*((z_t**2)/A_t)+b*(((z_t**2)/A_t)**2)
    kps2=c*((n_t-z_t)/(n_t+z_t))+d*(((n_t-z_t)/(n_t+z_t))**2)
    kps3=(e*Es)+f
    kps=kps1+kps2+kps3
    return kps+np.log10(3.155*10**7)
    
#Function for calculating Xu Model for SF half-lives
def xu(A_1,z_1,A_2,z_2,q):
    n_1=A_1-z_1
    n_2=A_2-z_2
    n_t=(n_1+n_2)
    z_t=(z_1+z_2)
    A_t=A_1+A_2 
    C0=-195.09227
    C1=3.10156
    C2=-0.04386
    C3=1.4030*(10**(-6))
    C4=-0.03199
    T1=C0+C1*A_t+C2*(z_t**2)+C3*(z_t**4)+C4*((n_t-z_t)**2)
    T2=0.13323*((z_t**2)/(A_t**(1/3.000)))-11.64
    return np.exp(2*pi*(T1-T2))
    
#Function for calculating CPPM for AD and CD half-lives
#If want to print out a,nu, turning points, P, lambda and logT values do 
#Uncomment commented parts in this function
def cppm(A_1,z_1,A_2,z_2,q):
    A_t=A_1+A_2
    Z_t=z_1+z_2
    N_t=A_t-Z_t
    b=1.00
    hbarc=197.327
    hbar=6.582*10**(-22)
    h=hbar*2*pi
    e=np.sqrt(1.44)
    mu=(931.502)*(A_1*A_2)/(A_1+A_2)
    def pot(z):
        if z<0:
            return phi1(z)
        else:
            gama=0.9517*(1-1.7826*((N_t-Z_t)**2)/(A_t**2))
            R1=1.28*(A_1**(1/3.00))-0.76+0.8*(A_1**(-1/3.00))
            R2=1.28*(A_2**(1/3.00))-0.76+0.8*(A_2**(-1/3.00))
            C1=R1-(b**2)/R1
            C2=R2-(b**2)/R2    
            Co=(C1*C2)/(C1+C2)
            x=z+C1+C2
            V_c=(z_1*z_2*e**2)/x
            V_p=4*pi*gama*b*Co*phi(z)
            V=V_c+V_p-q
            return V
    def Vpot(z):
            return pot(z)
    R1=1.28*(A_1**(1/3.00))-0.76+0.8*(A_1**(-1/3.00))
    R2=1.28*(A_2**(1/3.00))-0.76+0.8*(A_2**(-1/3.00))
    Rt=1.28*(A_t**(1/3.00))-0.76+0.8*(A_t**(-1/3.00))
    gama=0.9517*(1-1.7826*((N_t-Z_t)**2)/(A_t**2))    
    C1=R1-(b**2)/R1
    C2=R2-(b**2)/R2
    C=Rt-(b**2)/Rt    
    Co=(C1*C2)/(C1+C2)
    K=4*pi*gama*b*Co
    X1=-q+(z_1*z_2*e**2)/(C1+C2)+K*(-1.7817)
    X2=-(z_1*z_2*e**2)/((C1+C2)**2)+K*(0.9270)
    X3=2*(C1+C2-C)
    nu=(X2*X3)/X1
    a=X1/(X3**nu)
#    print "a=",a
#    print "n=",nu
    def phi1(z):
        R1=1.28*(A_1**(1/3.00))-0.76+0.8*(A_1**(-1/3.00))
        R2=1.28*(A_2**(1/3.00))-0.76+0.8*(A_2**(-1/3.00))
        Rt=1.28*(A_t**(1/3.00))-0.76+0.8*(A_t**(-1/3.00))
        C1=R1-(b**2)/R1
        C2=R2-(b**2)/R2
        C=Rt-(b**2)/Rt 
        Co=(C1*C2)/(C1+C2)
        K=4*pi*gama*b*Co
        X1=-q+(z_1*z_2*e**2)/(C1+C2)+K*(-1.7817)
        X2=-(z_1*z_2*e**2)/((C1+C2)**2.0)+K*(0.9270)
        X3=2*(C1+C2-C)
        nu=(X2*X3)/X1
        a=X1/(X3**nu)    
        L=z+2*(C1+C2)
        L0=2*C    
        return a*((L-L0)**nu)    
    def phi(z):
        if z>1.9475:
            return -4.41*np.exp(-z/0.7176)
        else:
            return -1.7817+0.9270*z+0.0169*(z**2.0)-0.05148*(z**3.0)          
    def k(z):
            cons=8*mu/((hbarc)**2.0)
            return np.sqrt(cons*np.abs(pot(z)))
    x_1=fsolve(Vpot,0.001)
    x_2=fsolve(Vpot,15.00)
    ans,err=integrate.quad(k,float(x_1),float(x_2))
    res=float(ans)
#   print "Turning points","a=",float(x_1),"b=",float(x_2)
    p=np.exp(-res)
#   print "Penetrability P=", p
    en=q*(0.056+0.039*np.exp((4.0-A_2)/2.5))
    nuu=2*en/h
#    print nuu
    lmb=nuu*p
    T_1=(np.log(2.0))/(lmb)
    LogT=np.log10(T_1)
#   print "Decay Constant lamda=",lmb

#   print "T_1/2 (s)=",T_1
#   print "LogT=",LogT
#    for z in np.arange(-20,50.00,0.1): 
#        P=pot(z)
#        x=z
#        y=P
#        plt.plot(x,y,"ro--")
#        plt.xlabel('z (fm)')
#        plt.ylabel('V-Q (MeV)')
#        #plt.legend(["$^{4}$He"])
#        plt.legend(["$^{12}$C"])
#        plt.axis([-5,20,0,30.0])
#        #plt.axis([-5,50,0,16.0])
#        plt.show()
    return LogT

#Function for calculating UDL half-lives
def udl(A_1,z_1,A_2,z_2,q):
    muu=(A_1*A_2)/(A_1+A_2)
#other version for parameters
#    a=0.3949
#    b=-0.3693
#    c=-23.7615
    a=0.4314
    b=-0.4087
    c=-25.7725
    ksi=z_1*z_2*(np.sqrt(muu/q))
    rhoo=np.sqrt(muu*z_1*z_2*(A_1**(1/3.00)+A_2**(1/3.00)))
    return a*ksi+b*rhoo+c
 
#Function for calculating Horoi half-lives
def horoi(A_1,z_1,A_2,z_2,q):
    mu=(A_1*A_2)/(A_1+A_2)
    a1=9.1
    b1=-10.2
    a2=7.39
    b2=-23.2
    x=0.416
    y=0.613
    kappa=(a1*mu**x+b1)*(((z_1*z_2)**y)/np.sqrt(q)-7.00)
    dig=(a2*mu**x)+b2
    return kappa+dig
#Function for calculating UNIV half-lives
def univ(A_1,z_1,A_2,z_2,q):
    mu=(A_1*A_2)/(A_1+A_2)
    f=-22.16917
    g=0.598*(A_2-1.00)
    Rb=(1.43998*z_1*z_2)/q
    Rt=1.2249*(A_1**(1/3.00)+A_2**(1/3.00))
    r00=Rt/Rb
    h=0.22873*np.sqrt(mu*z_1*z_2*Rb)*(np.arccos(np.sqrt(r00))-np.sqrt(r00*(1.00-r00)))
    return f+g+h

#Function for calculating Viola half-lives
def viola(A_1,z_1,A_2,z_2,q):
    a=1.66175
    b=-8.5166
    c=-0.20228
    d=-33.9069
    n_1=A_1-z_1
    n_2=A_2-z_2
    n_t=(n_1+n_2)
    z_t=(z_1+z_2)
    if z_t%2==0 and n_t%2==0:
        hlog=0.00
    elif z_t%2==1 and n_t%2==0:
        hlog=0.772
    elif z_t%2==0 and n_t%2==1:
        hlog=1.066
    else:
        hlog=1.114
    return (a*(z_1+z_2)+b)*(q**(-1/2.00))+c*(z_1+z_2)+d+hlog

#Function for calculating Royer half-lives
def royer(A_1,z_1,A_2,z_2,q):
    n_1=A_1-z_1
    n_2=A_2-z_2
    n_t=(n_1+n_2)
    z_t=(z_1+z_2)
    A_t=A_1+A_2    
    if z_t%2==0 and n_t%2==0:
        a=-25.31
        b=-1.1629
        c=1.5864
    elif z_t%2==0 and n_t%2==1:
        a=-26.65
        b=-1.0859
        c=1.5848
    elif z_t%2==1 and n_t%2==0:
        a=-25.68
        b=-1.1423
        c=1.5920
    else:
        a=-29.48
        b=-1.1130
        c=1.6971
    return a+b*((A_t)**(1/6.00))*np.sqrt(z_t)+(c*z_t)/(np.sqrt(q))

#Function for calculating SF with Soylu's formula

def soylu(A,z,n):
    C0=-10.0987592959
    C1=119.319858732
    C2=-0.516609881059
    C3=-9.52538327068
    C4=1.92155604207e-06
    C5=-1496.05967574
    T1=C0*(z+n)+C1*((z+n)**(2/3.000))+C2*(z*(z-1)*((z+n)**(-1/3.000)))
    T2=(C3*((n-z)**2))/(z+n)+C4*z**4+C5 
    T=np.exp(2.00*pi*(T1+T2))*(3.155*(10**7))
    return T


#Printing all values in terms of logarithmic do uncomment here and 
#comment the following printing part
#print "cppm=",calc(A_1,z_1,A_2,z_2,q)
#print "udl=",udl(A_1,z_1,A_2,z_2,q)
#print "horoi=",horoi(A_1,z_1,A_2,z_2,q)
#print "univ=",univ(A_1,z_1,A_2,z_2,q)
#print "viola=",viola(A_1,z_1,A_2,z_2,q)
#print "royer=", royer(A_1,z_1,A_2,z_2,q)     

print "Parent A=",A_1+A_2,"Parent Z=",z_1+z_2
print "SF(KPS)=",10**kps(A_1,z_1,A_2,z_2,q,Es),"s"
print "SF(Xu)=",xu(A_1,z_1,A_2,z_2,q),"s"
print "SF(Soylu)=",soylu(A,z,n),"s"
print "CPPM=",10**cppm(A_1,z_1,A_2,z_2,q),"s"
print "VSS=",10**viola(A_1,z_1,A_2,z_2,q),"s"
print "UNIV=",10**univ(A_1,z_1,A_2,z_2,q),"s"
print "Royer=", 10**royer(A_1,z_1,A_2,z_2,q),"s"   
print "UDL=",10**udl(A_1,z_1,A_2,z_2,q),"s"
print "Horoi=",10**horoi(A_1,z_1,A_2,z_2,q),"s"

#Calculate Branching Ratio: it compares the SF values of Xu and CPPM for alpha
#If want to compare SF values of KPS do uncomment the following

#CPPM for AD vs Xu for SF
br=cppm(A_1,z_1,A_2,z_2,q)-np.log10(xu(A_1,z_1,A_2,z_2,q))
if br<0.0:
    print "CPPM-Xu","BR=",br,"Mode=","AD"
else:
    print "CPPM-Xu","BR=",br,"Mode=","SF"

#UDL for AD vs Xu for SF
br1=udl(A_1,z_1,A_2,z_2,q)-np.log10(xu(A_1,z_1,A_2,z_2,q))
if br1<0.0:
    print "UDL-Xu","BR=",br1,"Mode=","AD"
else:
    print "UDL-Xu","BR=",br1,"Mode=","SF"

#CPPM for AD vs Soylu for SF
br=cppm(A_1,z_1,A_2,z_2,q)-np.log10(soylu(A,z,n))
if br<0.0:
    print "CPPM-Soylu","BR=",br,"Mode=","AD"
else:
    print "CPPM-Soylu","BR=",br,"Mode=","SF"  

#CPPM for AD vs KPS for SF
br=cppm(A_1,z_1,A_2,z_2,q)-(kps(A_1,z_1,A_2,z_2,q,Es))
if br<0.0:
    print "CPPM-KPS","BR=",br,"Mode=","AD"
else:
    print "CPPM-KPS","BR=",br,"Mode=","SF"  