"""
Created on Mon May 27 11:17:36 2024

@author: Giorgos Koufetidis
"""
import numpy as np
from scipy.integrate import quad,simpson
from sympy import symbols, pi, sqrt, log,exp, lambdify, oo,N,integrate
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])
elements = ["He","Li","Βe","B","C","N","O","F","Ne"]
Sr = []
Sk = []
S = []
Z= [2,3,4,5,6,7,8,9,10]
table = PrettyTable()
r,z,k = symbols("r z k")
sr_1s = 2*z**(3/2)*exp(-z*r)
sr_2s= 2/sqrt(3)*z**(5/2)*r*exp(-z*r)
sr_3s = 2**(3/2)/(3*sqrt(5))*z**(7/2)*r**2*exp(-z*r)
sr_4s=2/(3*sqrt(35)) * z**(9/2) * r**3 * exp(-z*r)
sr_5s=2**(3/2)/ (45*sqrt(7)) * z**(11/2) * r**4 * exp(-z*r)
sr_2p=2/sqrt(3)*z**(5/2)*r *exp(-z*r)
sr_3p=2**(3/2)/(3*sqrt(5))*z**(7/2)*r**2*exp(-z*r)
sr_4p=2/(3*sqrt(35)) * z**(9/2) * r**3 *exp(-z*r)
sr_5p=2**(3/2)/(45*sqrt(7)) * z**(11/2) * r**4 * exp(-z*r)
sr_3d=2**(3/2)/(3*sqrt(5)) * z**(7/2) * r**2 * exp(-z*r)
sr_4d=2/(3*sqrt(35)) * z**(9/2) * r**3 * exp(-z*r)

sk1s = 1/(2*pi)**(3/2)*16*pi*z**(5/2)/(z**2+k**2)**2
sk2s = 1/(2*pi)**(3/2)*16*pi*z**(5/2)*(3*z**2-k**2)/(sqrt(3)*(z**2+k**2)**3)
sk3s = 1/(2*pi)**(3/2)*64*sqrt(10)*pi*z**(9/2)*(z**2-k**2)/(5*(z**2+k**2)**4)
sk4s=  1/(2*pi)**(3/2)*64 *pi*z**(9/2)*(5*z**4-10*z**2*k**2 + k**4)/(sqrt(35)*(z**2 + k**2)**5)
sk5s=  1/(2*pi)**(3/2)*128*sqrt(14)*pi*z**(13/2)*(3* z**4 - 10* z**2* k**2 + 3*k**4)/(21*(z**2 + k**2)**6)
sk2p = 1/(2*pi)**(3/2)*64*pi*k*z**(7/2)/(sqrt(3)* (z**2 + k**2)**3)
sk3p = 1/(2*pi)**(3/2)*64*sqrt(10)* pi*k*z**(7/2)*(5*z**2-k**2) /(15*(z**2 + k**2)**4)
sk4p = 1/(2*pi)**(3/2)*128*pi*k*z**(11/2)*(5*z**2-3*k**2)/(sqrt(35)*(z**2 +k**2)**5)
sk5p = 1/(2*pi)**(3/2)*128*sqrt(14)*pi*k*z**(11/2)*(35*z**4 - 42*z**2* k**2 + 3*k**4)/(105*(z**2 + k**2)**6)
sk3d = 1/(2*pi)**(3/2)*128*sqrt(10)*pi*k**2*z**(9/2)/(5*(z**2+k**2)**4)
sk4d = 1/(2*pi)**(3/2)*128*pi*k**2*z**(9/2)*(7*z**2-k**2)/(sqrt(35)*(z**2+k**2)**5)

#He
print("He")
R1s_He = 1.3479*sr_1s.subs(z,1.4595) -0.001613*sr_3s.subs(z,5.3244) - 0.100506*sr_2s.subs(z,2.6298) - 0.270779*sr_2s.subs(z,1.7504) 
IR1_He = N(integrate(R1s_He**2*r**2,(r,0,oo)))
print("He:Normalization of R1s=",IR1_He)
pr_He = (1/(8*pi))* 2*R1s_He**2
I_pr_He = N(integrate(4*pi*pr_He*r**2,(r,0,oo)))
print("Ηe Normalization of pr: I=",I_pr_He)

NSr_He = lambdify(r,-4*pi*(pr_He*log(pr_He)*r**2),'numpy')
t= np.linspace(0,10,10000)
Sr_He = np.trapz(NSr_He(t),t)
Sr.append(round(Sr_He,4))
print()

#K
K1s_He= 1.3479*sk1s.subs(z,1.4595) -0.001613*sk3s.subs(z,5.3244) - 0.100506*sk2s.subs(z,2.6298) - 0.270779*sk2s.subs(z,1.7504)
N_K1s_He = lambdify(k,K1s_He**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_He = quad(N_K1s_He,0,np.inf)
print("He:Normalization of K1s=",Ik_He[0])
nk_He = 1/(8*pi) * (2*K1s_He**2)
I_nk_He = quad(lambdify(k,4*pi*nk_He*k**2),0,np.inf)
print("He:Normalization of nk=",I_nk_He[0])
NSk_He = lambdify(k,-4*pi*nk_He*log(nk_He)*k**2,'numpy')
Sk_He = quad(NSk_He,0,np.inf)
#print("He Sk=",Sk_He[0])
#print("He:Sr+Sk=",Sk_He[0]+ Sr_He)
print()
print("------------------")
print("")
Sk.append(round(Sk_He[0],4))
S.append(round(Sk_He[0]+Sr_He,4))
#Li
print("Li")
R1s_Li = 0.141279*sr_1s.subs(z,4.3069) + 0.874231*sr_1s.subs(z,2.4573)-0.005201*sr_3s.subs(z,6.785)-0.002307*sr_2s.subs(z,7.4527) + 0.006985*sr_2s.subs(z,1.8504)-0.000305*sr_2s.subs(z,0.7667) + 0.000760*sr_2s.subs(z,0.6364)
IR1_Li = N(integrate(R1s_Li**2*r**2,(r,0,oo)))
print("Li:Normalization of R1s=",IR1_Li)

R2s_Li = -0.022416*sr_1s.subs(z,4.3069) - 0.135791*sr_1s.subs(z,2.4573)+0.000389*sr_3s.subs(z,6.7850)-0.000068*sr_2s.subs(z,7.4527) - 0.076544*sr_2s.subs(z,1.8504)+0.340542*sr_2s.subs(z,0.7667) + 0.715708*sr_2s.subs(z,0.6364)
IR2_Li = N(integrate(R2s_Li**2*r**2,(r,0,oo)))
print("Li:Normalization of R2s=",IR2_Li)

pr_Li = 1/(4*pi*3)* (2*R1s_Li**2 + R2s_Li**2)
#pr_Li = (2*R1s_Li**2 + 2*R2s_Li**2)
I_pr_Li = N(integrate(4*pi*pr_Li*r**2,(r,0,oo)))
print("Li:Normalization of pr=",I_pr_Li)

NSr_Li = lambdify(r,-4*pi*pr_Li*log(pr_Li)*r**2,'numpy')
t1= np.linspace(0,100,1000)
Sr_Li =  simpson(NSr_Li(t1),t1)
print()
#K
print("K1s")
K1s_Li=  0.141279*sk1s.subs(z,4.3069) + 0.874231*sk1s.subs(z,2.4573)-0.005201*sk3s.subs(z,6.7850)-0.002307*sk2s.subs(z,7.4527) + 0.006985*sk2s.subs(z,1.8504)-0.000305*sk2s.subs(z,0.7667) + 0.000760*sk2s.subs(z,0.6364)
N_K1s_Li = lambdify(k,K1s_Li**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_Li = quad(N_K1s_Li,0,np.inf)
print("Li:Normalization of K1s=",Ik_Li[0])
print("")
print("K2s")
K2s_Li = -0.022416*sk1s.subs(z,4.3069) - 0.135791*sk1s.subs(z,2.4573)+0.000389*sk3s.subs(z,6.7850)-0.000068*sk2s.subs(z,7.4527) - 0.076544*sk2s.subs(z,1.8504)+0.340542*sk2s.subs(z,0.7667) + 0.715708*sk2s.subs(z,0.6364)
N_K2s_Li=lambdify(k,K2s_Li**2*k**2,'numpy' )
Ik2s_Li = quad(N_K2s_Li,0,np.inf)
print("Li:Normalization of K2s",Ik2s_Li[0])
print("")

nk_Li = 1/(4*pi*3) * (2*K1s_Li**2 + K2s_Li**2)

I_nk_Li = quad(lambdify(k,4*pi*nk_Li*k**2),0,np.inf)
print("Li:Normalization of nk=",I_nk_He[0])
NSk_Li = lambdify(k,-4*pi*nk_Li*log(nk_Li)*k**2,'numpy')
Sk_Li = quad(NSk_Li,0,np.inf)
#print("Li Sk=",Sk_Li[0])
#print("Li:Sr+Sk=",Sk_Li[0]+ Sr_Li)
Sk.append(round(Sk_Li[0],4))
Sr.append(round(Sr_Li,4))
S.append(round(Sk_Li[0]+Sr_Li,4))

#Be
print("Be")
R1s_Be = 0.285107*sr_1s.subs(z,5.7531) + 0.474813*sr_1s.subs(z,3.7156)-0.00162*sr_3s.subs(z,9.967)+0.052852*sr_3s.subs(z,3.7128) +0.243499* sr_2s.subs(z,4.4661)+0.000106*sr_2s.subs(z,1.2919) - 0.000032*sr_2s.subs(z,0.8555)
IR1_Be = N(integrate(R1s_Li**2*r**2,(r,0,oo)))
print("Be:Normalization of R1s=",IR1_Be)

R2s_Be = -0.016378*sr_1s.subs(z,5.7531) - 0.155066*sr_1s.subs(z,3.7156)+0.000426*sr_3s.subs(z,9.967)-0.059234*sr_3s.subs(z,3.7128) -0.031925* sr_2s.subs(z,4.4661)+0.387968*sr_2s.subs(z,1.2919) + 0.685674*sr_2s.subs(z,0.8555)
IR2_Be = N(integrate(R2s_Be**2*r**2,(r,0,oo)))
print("Be:Normalization of R2s=",IR2_Be)

pr_Be = 1/(4*pi*4)* (2*R1s_Be**2 + 2*R2s_Be**2)
#pr_Li = (2*R1s_Li**2 + 2*R2s_Li**2)
I_pr_Be = N(integrate(4*pi*pr_Be*r**2,(r,0,oo)))
print("Be:Normalization of pr=",I_pr_Be)

NSr_Be = lambdify(r,-4*pi*pr_Be*log(pr_Be)*r**2,'numpy')
Sr_Be =  simpson(NSr_Be(t1),t1)
print("Be Sr=",Sr_Be)
print()
#K
print("K1s")
K1s_Be= 0.285107*sk1s.subs(z,5.7531) + 0.474813*sk1s.subs(z,3.7156)-0.00162*sk3s.subs(z,9.967)+0.052852*sk3s.subs(z,3.7128) +0.243499* sk2s.subs(z,4.4661)+0.000106*sk2s.subs(z,1.2919) - 0.000032*sk2s.subs(z,0.8555) 
N_K1s_Be = lambdify(k,K1s_Be**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_Be = quad(N_K1s_Be,0,np.inf)
print("Be:Normalization of K1s=",Ik_Be[0])
print("")
print("K2s")
K2s_Be = -0.016378*sk1s.subs(z,5.7531) - 0.155066*sk1s.subs(z,3.7156)+0.000426*sk3s.subs(z,9.967)-0.059234*sk3s.subs(z,3.7128) -0.031925* sk2s.subs(z,4.4661)+0.387968*sk2s.subs(z,1.2919) + 0.685674*sk2s.subs(z,0.8555)
N_K2s_Be=lambdify(k,K2s_Be**2*k**2,'numpy' )
Ik2s_Be = quad(N_K2s_Be,0,np.inf)
print("Be:Normalization of K2s=",Ik2s_Be[0])
print("")
nk_Be = 1/(4*pi*4) * (2*K1s_Be**2+ 2*K2s_Be**2)

I_nk_Be = quad(lambdify(k,4*pi*nk_Be*k**2),0,np.inf)
print("nK of Be: I=",I_nk_Be[0])
NSk_Be = lambdify(k,-4*pi*nk_Be*log(nk_Be)*k**2,'numpy')
Sk_Be = quad(NSk_Be,0,np.inf)
#print("Li Sk=",Sk_Li[0])
#print("Li:Sr+Sk=",Sk_Li[0]+ Sr_Li)
Sk.append(round(Sk_Be[0],4))
Sr.append(round(Sr_Be,4))
S.append(round(Sk_Be[0]+Sr_Be,4))

#B
print("B")
R1s_B = 0.381607*sr_1s.subs(z,7.0178) +0.423958*sr_1s.subs(z,3.9468) - 0.001316*sr_3s.subs(z,12.7297) - 0.000822*sr_3s.subs(z,2.7646) + 0.237016*sr_2s.subs(z,5.7420) + 0.001062*sr_2s.subs(z,1.5436) - 0.000137*sr_2s.subs(z,1.0802)
IR1_B = N(integrate(R1s_B**2*r**2,(r,0,oo)))
print("B:Normalization of R1s=",IR1_B)
R2s_B = -0.022549*sr_1s.subs(z,7.0178) +0.321716*sr_1s.subs(z,3.9468) - 0.000452*sr_3s.subs(z,12.7297) - 0.072032*sr_3s.subs(z,2.7646) - 0.050313*sr_2s.subs(z,5.7420) - 0.484281*sr_2s.subs(z,1.5436) - 0.518986*sr_2s.subs(z,1.0802)
IR2s_B = N(integrate(R2s_B**2*r**2,(r,0,oo)))
print("B:Normalization of R2s=",IR2s_B)
R2p_B = 0.0076*sr_2p.subs(z,5.7416) + 0.045137*sr_2p.subs(z,2.6341)+0.184206*sr_2p.subs(z,1.834) + 0.394745*sr_2p.subs(z,1.1919) + 0.432795*sr_2p.subs(z,0.8494)
IR2p_B = N(integrate(R2p_B**2*r**2,(r,0,oo)))
print("B:Normalization of R2p=",IR2p_B)

pr_B = (1/(4*pi*5))* (2*R1s_B**2 + 2*R2s_B**2+ R2p_B**2)
I_pr_B = N(integrate(4*pi*pr_B*r**2,(r,0,oo)))
print("B:Normalization of pr I=",I_pr_B)
NSr_B = lambdify(r,-4*pi*(pr_B*log(pr_B)*r**2),'numpy')
Sr_B = np.trapz(NSr_B(t1),t1)
#print("He Sr=",Sr_He)
Sr.append(round(Sr_B,4))
print()
#K
K1s_B = 0.381607*sk1s.subs(z,7.0178) +0.423958*sk1s.subs(z,3.9468) - 0.001316*sk3s.subs(z,12.7297) - 0.000822*sk3s.subs(z,2.7646) + 0.237016*sk2s.subs(z,5.7420) + 0.001062*sk2s.subs(z,1.5436) - 0.000137*sk2s.subs(z,1.0802)
N_K1s_B = lambdify(k,K1s_B**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_B = quad(N_K1s_B,0,np.inf)
print("B:Normalization of K1s=",Ik_B[0])

K2s_B = -0.022549*sk1s.subs(z,7.0178) +0.321716*sk1s.subs(z,3.9468) - 0.000452*sk3s.subs(z,12.7297) - 0.072032*sk3s.subs(z,2.7646) - 0.050313*sk2s.subs(z,5.7420) - 0.484281*sk2s.subs(z,1.5436) - 0.518986*sk2s.subs(z,1.0802)
N_K2s_B = lambdify(k,K2s_B**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2s_B = quad(N_K2s_B,0,np.inf)
print("B:Normalization of K1s=",Ik2s_B[0])
K2p_B = 0.0076*sk2p.subs(z,5.7416) + 0.045137*sk2p.subs(z,2.6341)+0.184206*sk2p.subs(z,1.834) + 0.394745*sk2p.subs(z,1.1919) + 0.432795*sk2p.subs(z,0.8494)
N_K2p_B = lambdify(k,K2p_B**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2p_B = quad(N_K2p_B,0,np.inf)
print("B:Normalization of K2p=",Ik2p_B[0])

nk_B = 1/(4*pi*5) * (2*K1s_B**2+ 2*K2s_B**2+K2p_B**2)
I_nk_B = quad(lambdify(k,4*pi*nk_B*k**2),0,np.inf)
print("B:Normalization of nk: I=",I_nk_B[0])
NSk_B = lambdify(k,-4*pi*nk_B*log(nk_B)*k**2,'numpy')
Sk_B = quad(NSk_B,0,np.inf)
#print("He Sk=",Sk_He[0])
#print("He:Sr+Sk=",Sk_He[0]+ Sr_He)
print()
print("------------------")
print("")
Sk.append(round(Sk_B[0],4))
S.append(round(Sk_B[0]+Sr_B,4))

#C
print("C")
R1s_C = 0.352872*sr_1s.subs(z,8.4936) +0.473621*sr_1s.subs(z,4.8788) - 0.001199*sr_3s.subs(z,15.4660) + 0.210887*sr_2s.subs(z,7.05) + 0.000886*sr_2s.subs(z,2.264) + 0.000465*sr_2s.subs(z,1.4747) - 0.000119*sr_2s.subs(z,1.1639)
IR1_C = N(integrate(R1s_C**2*r**2,(r,0,oo)))
print("C:Normalization of R1s=",IR1_C)
R2s_C = -0.071727*sr_1s.subs(z,8.4936) + 0.438307*sr_1s.subs(z,4.8788) -0.000383*sr_3s.subs(z,15.4660) -0.091194*sr_2s.subs(z,7.05) -0.393105*sr_2s.subs(z,2.264) - 0.579121*sr_2s.subs(z,1.4747) - 0.126067*sr_2s.subs(z,1.1639)
IR2s_C = N(integrate(R2s_C**2*r**2,(r,0,oo)))
print("C:Normalization of R2s=",IR2s_C)
R2p_C = 0.006977*sr_2p.subs(z,7.05) + 0.070877*sr_2p.subs(z,3.2275)+0.230802*sr_2p.subs(z,2.1908) + 0.411931*sr_2p.subs(z,1.4413) + 0.350701*sr_2p.subs(z,1.0242)
IR2p_C = N(integrate(R2p_C**2*r**2,(r,0,oo)))
print("C:Normalization of R2p=",IR2p_C)

pr_C = (1/(4*pi*6))* (2*R1s_C**2+2*R2s_C**2+ 2*R2p_C**2)
I_pr_C = N(integrate(4*pi*pr_C*r**2,(r,0,oo)))
print("C:Normalization of pr I=",I_pr_C)
NSr_C = lambdify(r,-4*pi*(pr_C*log(pr_C)*r**2),'numpy')

Sr_C = np.trapz(NSr_C(t1),t1)
Sr.append(round(Sr_C,4))
print()
#K
K1s_C = 0.352872*sk1s.subs(z,8.4936) +0.473621*sk1s.subs(z,4.8788) - 0.001199*sk3s.subs(z,15.4660) + 0.210887*sk2s.subs(z,7.05) + 0.000886*sk2s.subs(z,2.264) + 0.000465*sk2s.subs(z,1.4747) - 0.000119*sk2s.subs(z,1.1639)
N_K1s_C = lambdify(k,K1s_C**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_C = quad(N_K1s_C,0,np.inf)
print("C:Normalization of K1s=",Ik_C[0])

K2s_C = -0.071727*sk1s.subs(z,8.4936) +0.438307*sk1s.subs(z,4.8788) -0.000383*sk3s.subs(z,15.4660) -0.091194*sk2s.subs(z,7.05) -0.393105*sk2s.subs(z,2.264) -0.579121*sk2s.subs(z,1.4747) - 0.126067*sk2s.subs(z,1.1639)

N_K2s_C = lambdify(k,K2s_C**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2s_C = quad(N_K2s_C,0,np.inf)
print("C:Normalization of K2s=",Ik2s_C[0])
K2p_C = 0.006977*sk2p.subs(z,7.05) + 0.070877*sk2p.subs(z,3.2275)+0.230802*sk2p.subs(z,2.1908) + 0.411931*sk2p.subs(z,1.4413) + 0.350701*sk2p.subs(z,1.0242)

N_K2p_C = lambdify(k,K2p_C**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2p_C = quad(N_K2p_C,0,np.inf)
print("C:Normalization of K2p=",Ik2p_C[0])


nk_C = 1/(4*pi*6) * (2*K1s_C**2+ 2*K2s_C**2+ 2*K2p_C**2)
I_nk_C = quad(lambdify(k,4*pi*nk_C*k**2),0,np.inf)
print("C:Normalization of nk: I=",I_nk_C[0])
NSk_C = lambdify(k,-4*pi*nk_C*log(nk_C)*k**2,'numpy')
Sk_C = quad(NSk_C,0,np.inf)
print()
print("------------------")
print("")
Sk.append(round(Sk_C[0],4))
S.append(round(Sk_C[0]+Sr_C,4))

#N

print("N")
R1s_N = 0.354839*sr_1s.subs(z,9.9051) + 0.472579*sr_1s.subs(z,5.7429) - 0.001038*sr_3s.subs(z,17.9816) + 0.208492*sr_2s.subs(z,8.3087) + 0.001687*sr_2s.subs(z,2.7611) + 0.000206*sr_2s.subs(z,1.8223) + 0.000064*sr_2s.subs(z,1.4191)
IR1_N = N(integrate(R1s_N**2*r**2,(r,0,oo)))
print("N:Normalization of R1s=",IR1_N)
R2s_N = -0.067498*sr_1s.subs(z,9.9051) + 0.434142*sr_1s.subs(z,5.7429) - 0.000315*sr_3s.subs(z,17.9816) -0.080331*sr_2s.subs(z,8.3087) - 0.374128*sr_2s.subs(z,2.7611) -0.522775*sr_2s.subs(z,1.8223) - 0.207735*sr_2s.subs(z,1.4191)
IR2s_N = N(integrate(R2s_N**2*r**2,(r,0,oo)))
print("N:Normalization of R2s=",IR2s_N)
R2p_N = 0.006323*sr_2p.subs(z,8.349) + 0.082938*sr_2p.subs(z,3.8827)+ 0.260147*sr_2p.subs(z,2.592) + 0.418361*sr_2p.subs(z,1.6946) + 0.308272*sr_2p.subs(z,1.1914)
IR2p_N = N(integrate(R2p_N**2*r**2,(r,0,oo)))
print("N:Normalization of R2p=",IR2p_N)

pr_N = (1/(4*pi*7))* (2*R1s_N**2+2*R2s_N**2+ 3*R2p_N**2)
I_pr_N = N(integrate(4*pi*pr_N*r**2,(r,0,oo)))
print("N:Normalization of pr I=",I_pr_N)

NSr_N = lambdify(r,-4*pi*(pr_N*log(pr_N)*r**2),'numpy')
Sr_N = np.trapz(NSr_N(t1),t1)
#print("He Sr=",Sr_He)
Sr.append(round(Sr_N,4))
print()
#N
K1s_N = 0.354839*sk1s.subs(z,9.9051) + 0.472579*sk1s.subs(z,5.7429) - 0.001038*sk3s.subs(z,17.9816) + 0.208492*sk2s.subs(z,8.3087) + 0.001687*sk2s.subs(z,2.7611) + 0.000206*sk2s.subs(z,1.8223) + 0.000064*sk2s.subs(z,1.4191)
N_K1s_N = lambdify(k,K1s_N**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_N = quad(N_K1s_N,0,np.inf)
print("N:Normalization of K1s=",Ik_N[0])

K2s_N = -0.067498*sk1s.subs(z,9.9051) + 0.434142*sk1s.subs(z,5.7429) - 0.000315*sk3s.subs(z,17.9816) -0.080331*sk2s.subs(z,8.3087) - 0.374128*sk2s.subs(z,2.7611) -0.522775*sk2s.subs(z,1.8223) - 0.207735*sk2s.subs(z,1.4191)

N_K2s_N = lambdify(k,K2s_N**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2s_N = quad(N_K2s_N,0,np.inf)
print("N:Normalization of K2s=",Ik2s_N[0])

K2p_N = 0.006323*sk2p.subs(z,8.349) + 0.082938*sk2p.subs(z,3.8827)+ 0.260147*sk2p.subs(z,2.592) + 0.418361*sk2p.subs(z,1.6946) + 0.308272*sk2p.subs(z,1.1914)
N_K2p_N = lambdify(k,K2p_N**2*k**2,'numpy')
Ik2p_N = quad(N_K1s_N,0,np.inf)
print("N:Normalization of K2p=",Ik2p_N[0])


nk_N = 1/(4*pi*7) * (2*K1s_N**2+ 2*K2s_N**2+ 3*K2p_N**2)
I_nk_N = quad(lambdify(k,4*pi*nk_N*k**2),0,np.inf)
print("N:Normalization of nk: I=",I_nk_N[0])
NSk_N = lambdify(k,-4*pi*nk_N*log(nk_N)*k**2,'numpy')
Sk_N = quad(NSk_N,0,np.inf)
print()
print("------------------")
print("")
Sk.append(round(Sk_N[0],4))
S.append(round(Sk_N[0]+Sr_N,4))

#O
print("O")
R1s_O = 0.360063*sr_1s.subs(z,11.297) + 0.466625*sr_1s.subs(z,6.5966) - 0.000918*sr_3s.subs(z,20.5019) + 0.208441*sr_2s.subs(z,9.5546) + 0.002018*sr_2s.subs(z,3.2482) + 0.000216*sr_2s.subs(z,2.1608) + 0.000133*sr_2s.subs(z,1.6411)
IR1_O = N(integrate(R1s_O**2*r**2,(r,0,oo)))
print("O:Normalization of R1s=",IR1_O)
R2s_O = -0.064363*sr_1s.subs(z,11.297) + 0.433186*sr_1s.subs(z,6.5966) - 0.000275*sr_3s.subs(z,20.5019) -0.072497*sr_2s.subs(z,9.5546) -0.3699*sr_2s.subs(z,3.2482) -0.512627*sr_2s.subs(z,2.1608) - 0.227421*sr_2s.subs(z,1.6411)
IR2s_O = N(integrate(R2s_O**2*r**2,(r,0,oo)))
print("O:Normalization of R2s=",IR2s_O)
R2p_O = 0.005626*sr_2p.subs(z,9.6471) + 0.126618*sr_2p.subs(z,4.3323)+ 0.328966*sr_2p.subs(z,2.7502) + 0.395422*sr_2p.subs(z,1.7525) + 0.231788*sr_2p.subs(z,1.2473)
IR2p_O = N(integrate(R2p_O**2*r**2,(r,0,oo)))
print("O:Normalization of R2p=",IR2p_O)

pr_O = (1/(4*pi*8))* (2*R1s_O**2+ 2*R2s_O**2+ 4*R2p_O**2)
I_pr_O = N(integrate(4*pi*pr_O*r**2,(r,0,oo)))
print("O:Normalization of pr I=",I_pr_O)

NSr_O = lambdify(r,-4*pi*(pr_O*log(pr_O)*r**2),'numpy')

Sr_O = np.trapz(NSr_O(t1),t1)
#print("He Sr=",Sr_He)
Sr.append(round(Sr_O,4))

print()
K1s_O = 0.360063*sk1s.subs(z,11.297) + 0.466625*sk1s.subs(z,6.5966) - 0.000918*sk3s.subs(z,20.5019) + 0.208441*sk2s.subs(z,9.5546) + 0.002018*sk2s.subs(z,3.2482) + 0.000216*sk2s.subs(z,2.1608) + 0.000133*sk2s.subs(z,1.6411)
N_K1s_O = lambdify(k,K1s_N**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_O = quad(N_K1s_N,0,np.inf)
print("O:Normalization of K1s=",Ik_O[0])

K2s_O = -0.064363*sk1s.subs(z,11.297) + 0.433186*sk1s.subs(z,6.5966) - 0.000275*sk3s.subs(z,20.5019) -0.072497*sk2s.subs(z,9.5546) -0.3699*sk2s.subs(z,3.2482) -0.512627*sk2s.subs(z,2.1608) - 0.227421*sk2s.subs(z,1.6411)
N_K2s_O = lambdify(k,K2s_O**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2s_O = quad(N_K2s_O,0,np.inf)
print("O:Normalization of K2s=",Ik2s_O[0])

K2p_O = 0.005626*sk2p.subs(z,9.6471) + 0.126618*sk2p.subs(z,4.3323)+ 0.328966*sk2p.subs(z,2.7502) + 0.395422*sk2p.subs(z,1.7525) + 0.231788*sk2p.subs(z,1.2473)
Ik2p_O = quad(N_K1s_O,0,np.inf)
print("O:Normalization of K2p=",Ik2p_O[0])

nk_O = 1/(4*pi*8) * (2*K1s_O**2+ 2*K2s_O**2+ 4*K2p_O**2)
I_nk_O = quad(lambdify(k,4*pi*nk_O*k**2),0,np.inf)
print("O:Normalization of nk: I=",I_nk_O[0])
NSk_O = lambdify(k,-4*pi*nk_O*log(nk_O)*k**2,'numpy')
Sk_O = quad(NSk_O,0,np.inf)

print()
print("------------------")
print("")
Sk.append(round(Sk_O[0],4))
S.append(round(Sk_O[0]+Sr_O,4))

#F
print("F")
R1s_F = 0.377498*sr_1s.subs(z,12.6074) + 0.443947*sr_1s.subs(z,7.4101) - 0.000797*sr_3s.subs(z,23.2475) + 0.213846*sr_2s.subs(z,10.7416) + 0.002183*sr_2s.subs(z,3.7543) + 0.000335*sr_2s.subs(z,2.5009) + 0.000147*sr_2s.subs(z,1.8577)
IR1_F = N(integrate(R1s_F**2*r**2,(r,0,oo)))
print("F:Normalization of R1s=",IR1_F)
R2s_F = -0.058489*sr_1s.subs(z,12.6074) + 0.42645*sr_1s.subs(z,7.4101) - 0.000274*sr_3s.subs(z,23.2475) -0.063457*sr_2s.subs(z,10.7416) -0.358939*sr_2s.subs(z,3.7543) -0.51666*sr_2s.subs(z,2.5009) -0.239143*sr_2s.subs(z,1.8577)
IR2s_F = N(integrate(R2s_F**2*r**2,(r,0,oo)))
print("F:Normalization of R2s=",IR2s_F)
R2p_F = 0.004879*sr_2p.subs(z,11.0134) +0.130794*sr_2p.subs(z,4.9962)+ 0.337876*sr_2p.subs(z,3.154) + 0.396122*sr_2p.subs(z,1.9722) + 0.225374*sr_2p.subs(z,1.3632)
IR2p_F = N(integrate(R2p_F**2*r**2,(r,0,oo)))
print("F:Normalization of R2p=",IR2p_F)

pr_F = (1/(4*pi*9))* (2*R1s_F**2+ 2*R2s_F**2+ 5*R2p_F**2)
I_pr_F = N(integrate(4*pi*pr_F*r**2,(r,0,oo)))
print("F:Normalization of pr I=",I_pr_F)

NSr_F = lambdify(r,-4*pi*(pr_F*log(pr_F)*r**2),'numpy')

Sr_F = np.trapz(NSr_F(t1),t1)
#print("He Sr=",Sr_He)
Sr.append(round(Sr_F,4))

print()
K1s_F = 0.377498*sk1s.subs(z,12.6074) + 0.443947*sk1s.subs(z,7.4101) - 0.000797*sk3s.subs(z,23.2475) + 0.213846*sk2s.subs(z,10.7416) + 0.002183*sk2s.subs(z,3.7543) + 0.000335*sk2s.subs(z,2.5009) + 0.000147*sk2s.subs(z,1.8577)
N_K1s_F = lambdify(k,K1s_F**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_F = quad(N_K1s_F,0,np.inf)
print("F:Normalization of K1s=",Ik_F[0])

K2s_F = -0.058489*sk1s.subs(z,12.6074) + 0.42645*sk1s.subs(z,7.4101) - 0.000274*sk3s.subs(z,23.2475) -0.063457*sk2s.subs(z,10.7416) -0.358939*sk2s.subs(z,3.7543) -0.51666*sk2s.subs(z,2.5009) -0.239143*sk2s.subs(z,1.8577)
N_K2s_F = lambdify(k,K2s_F**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2s_F = quad(N_K2s_F,0,np.inf)
print("F:Normalization of K2s=",Ik2s_F[0])

K2p_F = 0.004879*sk2p.subs(z,11.0134) +0.130794*sk2p.subs(z,4.9962)+ 0.337876*sk2p.subs(z,3.154) + 0.396122*sk2p.subs(z,1.9722) + 0.225374*sk2p.subs(z,1.3632)
NK2p_F = lambdify(k,K2p_F**2*k**2,'numpy')
IK2p_F = quad(NK2p_F,0,np.inf)
print("F:Normalization of K2p=",IK2p_F[0])

nk_F = 1/(4*pi*9) * (2*K1s_F**2+ 2*K2s_F**2+ 5*K2p_F**2)
I_nk_F = quad(lambdify(k,4*pi*nk_F*k**2),0,np.inf)
print("F:Normalization of nk: I=",I_nk_F[0])
NSk_F = lambdify(k,-4*pi*nk_F*log(nk_F)*k**2,'numpy')
Sk_F = quad(NSk_F,0,np.inf)
#print("He Sk=",Sk_He[0])
#print("He:Sr+Sk=",Sk_He[0]+ Sr_He)
print()
print("------------------")
print("")
Sk.append(round(Sk_F[0],4))
S.append(round(Sk_F[0]+Sr_F,4))

#Ne

print("Ne")
R1s_Ne = 0.392290*sr_1s.subs(z,13.9074) +  0.425817*sr_1s.subs(z,8.2187) -0.000702*sr_3s.subs(z,26.0325) +  0.217206 *sr_2s.subs(z,11.9249) +0.002300 *sr_2s.subs(z,4.2635) +  0.000463*sr_2s.subs(z,2.8357) +  0.000147*sr_2s.subs(z,2.0715)
IR1_Ne = N(integrate(R1s_Ne**2*r**2,(r,0,oo)))
print("Ne:Normalization of R1s=",IR1_Ne)
R2s_Ne = -0.053023*sr_1s.subs(z,13.9074) + 0.419502*sr_1s.subs(z,8.2187) -0.000263*sr_3s.subs(z,26.0325) - 0.055723*sr_2s.subs(z,11.9249) -0.349457*sr_2s.subs(z,4.2635) - 0.523070*sr_2s.subs(z,2.8357) - 0.246038*sr_2s.subs(z,2.0715)
IR2s_Ne = N(integrate(R2s_Ne**2*r**2,(r,0,oo)))
print("Ne:Normalization of R2s=",IR2s_Ne)
R2p_Ne = 0.004391*sr_2p.subs(z,12.3239) +0.133955*sr_2p.subs(z,5.6525)+ 0.342978*sr_2p.subs(z,3.557) + 0.395742*sr_2p.subs(z,2.2056) + 0.221831*sr_2p.subs(z,1.4948)
IR2p_Ne = N(integrate(R2p_Ne**2*r**2,(r,0,oo)))
print("Ne:Normalization of R2p=",IR2p_Ne)

pr_Ne = (1/(4*pi*10))* (2*R1s_Ne**2+ 2*R2s_Ne**2+ 6*R2p_Ne**2)
I_pr_Ne = N(integrate(4*pi*pr_Ne*r**2,(r,0,oo)))
print("Ne:Normalization of pr I=",I_pr_Ne)

NSr_Ne = lambdify(r,-4*pi*(pr_Ne*log(pr_Ne)*r**2),'numpy')

Sr_Ne = np.trapz(NSr_Ne(t1),t1)
#print("He Sr=",Sr_He)
Sr.append(round(Sr_Ne,4))

print()
K1s_Ne = 0.392290*sk1s.subs(z,13.9074) +  0.425817*sk1s.subs(z,8.2187) -0.000702*sk3s.subs(z,26.0325) +  0.217206 *sk2s.subs(z,11.9249) +0.002300 *sk2s.subs(z,4.2635) +  0.000463*sk2s.subs(z,2.8357) +  0.000147*sk2s.subs(z,2.0715)
N_K1s_Ne = lambdify(k,K1s_Ne**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik_Ne = quad(N_K1s_Ne,0,np.inf)
print("Ne:Normalization of K1s=",Ik_Ne[0])

K2s_Ne = -0.053023*sk1s.subs(z,13.9074) + 0.419502*sk1s.subs(z,8.2187) -0.000263*sk3s.subs(z,26.0325) - 0.055723*sk2s.subs(z,11.9249) -0.349457*sk2s.subs(z,4.2635) - 0.523070*sk2s.subs(z,2.8357) - 0.246038*sk2s.subs(z,2.0715)
N_K2s_Ne = lambdify(k,K2s_Ne**2*k**2,'numpy' ) #Converts a symbolic expression to a numerical function
Ik2s_Ne = quad(N_K2s_Ne,0,np.inf)
print("Ne:Normalization of K2s=",Ik2s_Ne[0])

K2p_Ne = 0.004391*sk2p.subs(z,12.3239) +0.133955*sk2p.subs(z,5.6525)+ 0.342978*sk2p.subs(z,3.557) + 0.395742*sk2p.subs(z,2.2056) + 0.221831*sk2p.subs(z,1.4948)
NK2p_Ne = lambdify(k,K2p_Ne**2*k**2,'numpy')
IK2p_Ne = quad(NK2p_Ne,0,np.inf)
print("Ne:Normalization of K2p=",IK2p_Ne[0])

nk_Ne = 1/(4*pi*10) * (2*K1s_Ne**2+ 2*K2s_Ne**2+ 6*K2p_Ne**2)
I_nk_Ne = quad(lambdify(k,4*pi*nk_Ne*k**2),0,np.inf)
print("Ne:Normalization of nk: I=",I_nk_Ne[0])
NSk_Ne = lambdify(k,-4*pi*nk_Ne*log(nk_Ne)*k**2,'numpy')
Sk_Ne = quad(NSk_Ne,0,np.inf)
#print("He Sk=",Sk_He[0])
#print("He:Sr+Sk=",Sk_He[0]+ Sr_He)
print()
print("------------------")
print("")
Sk.append(round(Sk_Ne[0],4))
S.append(round(Sk_Ne[0]+Sr_Ne,4))


table.add_column("Z",Z)
table.add_column("Atom",elements)
table.add_column("Sr",Sr)
table.add_column("Sk",Sk)
table.add_column("S",S)
print(table)


#R1s_Ne = 0.392290*sr_1s.subs(z,13.9074) + 0.425817 *sr_1s.subs(z,8.2187) -0.000702* sr_3s.subs(z,26.0325) + 0.217206*sr_2s.subs(z,11.9249) + 0.002300 *sr_2s.subs(z,4.2635) + 0.000463*sr_2s.subs(z,2.8357) + 0.000147* sr_2s.subs(z,2.0715)
#R1s_integration=N(integrate(R1s_Ne**2*r**2,(r,0,oo)))
#print(" Integration of R1s^2*r^2 is =",R1s_integration)

#R2s = -0.053023*sr_1s.subs(z,13.9074) + 0.419502*sr_1s.subs(z,8.2187)- 0.000263*sr_3s.subs(z,26.0325)- 0.055723*sr_2s.subs(z,11.9249) -0.349457*sr_2s.subs(z,4.2635) - 0.523070*sr_2s.subs(z,2.8357) - 0.246038*sr_2s.subs(z,2.0715)
#R2s_integration=N(integrate(R2s**2*r**2,(r,0,oo)))
#print(" Integration of R2s^2*r^2 is =",R2s_integration)
#R2p = 0.004391*sr_2p.subs(z,12.3239) + 0.133955*sr_2p.subs(z,5.6525) + 0.342978* sr_2p.subs(z,3.5570) + 0.395742*sr_2p.subs(z,2.2056) + 0.221831*sr_2p.subs(z,1.4948)
#R2p_integration = N(integrate(R2p**2*r**2,(r,0,oo)))



#pr = 1/(40*pi) * (2*R1s_Ne**2+2*R2s**2+6*R2p**2)
#pr_integration=N(4*pi*N(integrate(pr*r**2,(r,0,oo))))
#SrNe = -4*pi*N(integrate(pr*log(pr)*r**2,(r,0,oo)))






"""
# Define the function to integrate
def integrand(k):
    K1s_val = (
        0.392290 * (1/(2*np.pi)**(3/2)*16*np.pi*13.9074**(5/2)/(13.9074**2+k**2)**2) +
        0.425817 * (1/(2*np.pi)**(3/2)*16*np.pi*8.2187**(5/2)/(8.2187**2+k**2)**2) -
        0.000702 * (1/(2*np.pi)**(3/2)*64*np.sqrt(10)*np.pi*26.0325**(9/2)*(26.0325**2-k**2)/(5*(26.0325**2+k**2)**4)) +
        0.217206 * (1/(2*np.pi)**(3/2)*16*8.2187**(5/2)*(3*11.9249**2-k**2)/(np.sqrt(3)*(11.9249**2+k**2)**3)) +
        0.002300 * (1/(2*np.pi)**(3/2)*16*np.pi*4.2635**(5/2)/(4.2635**2+k**2)**2) +
        0.000463 * (1/(2*np.pi)**(3/2)*16*np.pi*2.8357**(5/2)/(2.8357**2+k**2)**2) +
        0.000147 * (1/(2*np.pi)**(3/2)*16*np.pi*2.0715**(5/2)/(2.0715**2+k**2)**2)
    )
    return K1s_val**2 * k**2

# Perform the numerical integration
K1s_integration_num, _ = quad(integrand, 0, np.inf)

print("Numerical integration of K1s^2*k^2 is =", K1s_integration_num)

"""
K1s = 0.392290* sk1s.subs(z,13.9074) + 0.425817* sk1s.subs(z,8.2187) -0.000702 *sk3s.subs(z,26.0325) + 0.217206* sk2s.subs(z,11.9249) +0.002300* sk2s.subs(z,4.2635) + 0.000463* sk2s.subs(z,2.8357) + 0.000147* sk2s.subs(z,2.0715)
#K1s_integration = N(integrate(K1s**2*k**2,(k,0,oo)))
K2s = -0.053023* sk1s.subs(z,13.9074) + 0.419502* sk1s.subs(z,8.2187) -0.000263*sk3s.subs(z,26.0325) - 0.055723* sk2s.subs(z,11.9249) -0.349457* sk2s.subs(z,4.2635) - 0.523070*sk2s.subs(z,2.8357) - 0.246038* sk2s.subs(z,2.0715)
K2p = 0.004391*sk2p.subs(z,12.3239) + 0.133955*sk2p.subs(z,5.6525) +0.342978*sk2p.subs(z,3.5570) + 0.395742*sk2p.subs(z,2.2056) + 0.221831*sk2p.subs(z,1.4948)
nk = (1/40*pi) * (2*K1s**2+2*K2s**2 + 6*K2p**2)

#Ploting Sk
plt.plot(Z,Sk,".-")
plt.xlabel("Z")
plt.ylabel("Sk")
plt.title("Entropy in momentum space")
plt.grid()
plt.show()

plt.plot(Z,Sr,".-")
plt.xlabel("Z")
plt.ylabel("Sr")
plt.title("Entropy in position space")
plt.grid()
plt.show()

plt.plot(Z,S,".-")
plt.xlabel("Z")
plt.ylabel("S")
plt.title("Information Entropy")
plt.grid()
plt.show()










#Radial distrubution function
t = np.linspace(0,4,1000)
R1sHe = lambdify(r,r**2*R1s_He**2)

R1sLi = lambdify(r,r**2*R1s_Li**2)
R2sLi = lambdify(r,r**2*R2s_Li**2)

R1sBe = lambdify(r,r**2*R1s_Be**2)
R2sBe = lambdify(r,r**2*R2s_Be**2)

R1sB = lambdify(r,r**2*R1s_B**2)
R2sB = lambdify(r,r**2*R2s_B**2)
R2pB = lambdify(r,r**2*R2p_B**2)

R1sC = lambdify(r,r**2*R1s_C**2)
R2sC = lambdify(r,r**2*R2s_C**2)
R2pC = lambdify(r,r**2*R2p_C**2)

R1sN = lambdify(r,r**2*R1s_N**2)
R2sN = lambdify(r,r**2*R2s_N**2)
R2pN = lambdify(r,r**2*R2p_N**2)

R1sO = lambdify(r,r**2*R1s_O**2)
R2sO = lambdify(r,r**2*R2s_O**2)
R2pO = lambdify(r,r**2*R2p_O**2)

R1sF = lambdify(r,r**2*R1s_F**2)
R2sF = lambdify(r,r**2*R2s_F**2)
R2pF = lambdify(r,r**2*R2p_F**2)

R1sNe = lambdify(r,r**2*R1s_Ne**2)
R2sNe= lambdify(r,r**2*R2s_Ne**2)
R2pNe = lambdify(r,r**2*R2p_Ne**2)


plt.plot(t,R1sHe(t),label="He")
plt.plot(t,R1sLi(t),label="Li")
plt.plot(t,R1sBe(t),label="Be")
plt.plot(t,R1sB(t),label="B")
plt.plot(t,R1sC(t),label="C")
plt.plot(t,R1sN(t),label="N")
plt.plot(t,R1sO(t),label="O")
plt.plot(t,R1sF(t),label="F")
plt.plot(t,R1sNe(t),label="Ne")
plt.title(r"$R1s^2r^2$")
plt.xlabel("r")
plt.grid()
plt.legend()
plt.show()

t = np.linspace(0,8,1000)
plt.plot(t,R2sLi(t),label="Li")
plt.plot(t,R2sBe(t),label="Be")
plt.plot(t,R2sB(t),label="B")
plt.plot(t,R2sC(t),label="C")
plt.plot(t,R2sN(t),label="N")
plt.plot(t,R2sO(t),label="O")
plt.plot(t,R2sF(t),label="F")
plt.plot(t,R2sNe(t),label="Ne")
plt.title(r"$R2s^2r^2$")
plt.xlabel("r(A)")
plt.grid()
plt.legend()
plt.show()


t = np.linspace(0,6,1000)
plt.plot(t,R2pB(t),label="B")
plt.plot(t,R2pC(t),label="C")
plt.plot(t,R2pN(t),label="N")
plt.plot(t,R2pO(t),label="O")
plt.plot(t,R2pF(t),label="F")
plt.plot(t,R2pNe(t),label="Ne")
plt.title(r"$R2p^2r^2$")
plt.xlabel("r(A)")
plt.grid()
plt.legend()
plt.show()








