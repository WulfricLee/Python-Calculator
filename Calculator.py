#!/usr/bin/python

# Set linewidth display 
# https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.set_printoptions.html
np.set_printoptions(linewidth = 200)

# # Which is used to turn ON/OFF debug
# import pdb
# _DEBUG = True

# # Watch all created variables
# dir()

############################

# ===== To use SymPy ===== #
from sympy import *
# Deg-Rad transform functions
def toDeg(x):
	x = (x/pi * 180).evalf()
	return x

def toRad(x):
	x = x/180 * pi
	return x

# Display complex in abs<deg form.
def toAbsDeg(x):
	ab = abs(x)
	deg = toDeg(arg(x))
	print(str(ab) + '\n' + "<" + str(deg))

# Convert abs<deg form to real+imag*I form.
def toRealImag(ab, de):
	return (ab*cos(toRad(de)) + ab*sin(toRad(de))*I).evalf()

# from mpmath import *
# mp.dps = 15; mp.pretty = True

# Symbolize English alphabet
# E, I were predefined.
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,F,G,H,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z = \
symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D F G H J K L M N O P Q R S T U V W X Y Z')
x = symbols('x')

# Commonly used Greek alphabet
uDef_alpha = symbols('uDef_alpha')
uDef_beta = symbols('uDef_beta')
uDef_gamma = symbols('uDef_gamma')
uDef_delta = symbols('uDef_delta')
uDef_epsilon = symbols('uDef_epsilon')
uDef_zeta = symbols('uDef_zeta')
uDef_eta = symbols('uDef_eta')
uDef_theta = symbols('uDef_theta')
uDef_iota = symbols('uDef_iota')
uDef_kappa = symbols('uDef_kappa')
uDef_lambda = symbols('uDef_lambda')
uDef_mu = symbols('uDef_mu')
uDef_nu = symbols('uDef_nu')
uDef_xi = symbols('uDef_xi')
uDef_omicron = symbols('uDef_omicron')
uDef_pi = symbols('uDef_pi')
uDef_rho = symbols('uDef_rho')
uDef_sigma = symbols('uDef_sigma')
uDef_tau = symbols('uDef_tau')
uDef_upsilon = symbols('uDef_upsilon')
uDef_phi = symbols('uDef_phi')
uDef_chi = symbols('uDef_chi')
uDef_psi = symbols('uDef_psi')
uDef_omega = symbols('uDef_omega')
uDef_varphi = symbols('uDef_varphi')
uDef_Alpha = symbols('uDef_Alpha')
uDef_Beta = symbols('uDef_Beta')
uDef_Gamma = symbols('uDef_Gamma')
uDef_Delta = symbols('uDef_Delta')
uDef_Epsilon = symbols('uDef_Epsilon')
uDef_Zeta = symbols('uDef_Zeta')
uDef_Eta = symbols('uDef_Eta')
uDef_Theta = symbols('uDef_Theta')
uDef_Iota = symbols('uDef_Iota')
uDef_Kappa = symbols('uDef_Kappa')
uDef_Lambda = symbols('uDef_Lambda')
uDef_Mu = symbols('uDef_Mu')
uDef_Nu = symbols('uDef_Nu')
uDef_Xi = symbols('uDef_Xi')
uDef_Omicron = symbols('uDef_Omicron')
uDef_Pi = symbols('uDef_Pi')
uDef_Rho = symbols('uDef_Rho')
uDef_Sigma = symbols('uDef_Sigma')
uDef_Tau = symbols('uDef_Tau')
uDef_Upsilon = symbols('uDef_Upsilon')
uDef_Phi = symbols('uDef_Phi')
uDef_Chi = symbols('uDef_Chi')
uDef_Psi = symbols('uDef_Psi')
uDef_Omega = symbols('uDef_Omega')
uDef_Varphi = symbols('uDef_Varphi')

# Display result in fractional form, use key word "tolerance=0.xxx" to specify accuracies.
nsimplyfy(expression)
nsimplify(0.33333, tolerance=1e-4)
nsimplify(pi, tolerance=0.01)
# Derivatives(differentiation)
diff(x**3, x)
# Integrate
integrate(sin(x), x)
# Use integrate(f, (x,a,b)) to compute the definite integrals
integrate(x**3, (x,0,1))  # the area under x^3 from x=0 to x=1
# Exponential
exp(x) # same as E**x
# Get exact value evalf()
E.evalf()
pi.evalf()
# Simplify an expression
simplify(expression)
# Factoring an expression
factor(expression)
# Expand an expression
expand(expression)
# Solve a equation(expression must equals to zero)
solve(expression, x)
# Complex number operations
# normal form: 
(a + b*I)
# polar form: 
abs*exp_polar(I*angle)
# operations:
re(expression)
im(expression)
conjugate(expression)
abs(expression)
arg(expression)

# === Matrix ===
M = Matrix([
	[1, -1], 
	[3, 4], 
	[0, 2]
	])
# Matrix Shape
M.shape
# Inverse Matrix
M**-1
# Transpose of a Matrix
M.T
# Accessing Rows and Columns
M.row(0)
M.col(2)
# Deleting and Inserting Rows and Columns
M.col_del(0)
M.row_del(1)
M = M.row_insert(1, Matrix([[0, 4]]))
M = M.col_insert(0, Matrix([1, -2]))
# Matrix Constructors
eye(3)
# ==============

# === Vector ===
# A 3D Cartesian coordinate system can be initialized in sympy.vector as
from sympy.vector import CoordSys3D
N = CoordSys3D('N')
# Once a coordinate system (in essence, a CoordSys3D instance) has been defined, we can access the orthonormal unit vectors 
# (i.e. the i, j and k vectors) and coordinate variables/base scalars (i.e. the x, y and z variables) corresponding to it. 
# N.i
# N.j
# N.k
# are unit vectors
# 
# To define a vector
v = 1 *N.i + 1 *N.j + 1 *N.k
# length
v.magnitude()
# unitized
v.normalize()
# dot product
v1.dot(v2)
# crose product
v1.crose(v2)
# vector to matrix
v.to_matrix(N)
# ==============

# Sientific expression
#2e-2 = 0.02
#2e-3 = 0.002

############################

# ===== To use NumPy ===== #
import numpy as np
# Deg-Rad transform functions
def nptoDeg(x):
	x = x/np.pi * 180
	return x
def nptoRad(x):
	x = x/180 * np.pi
	return x
# Display complex in abs<deg form.
def nptoAbsDeg(x):
	abs = np.absolute(x)
	deg = nptoDeg(np.angle(x))
	print(str(abs) + '\n' + "<" + str(deg))
# Pi
np.pi
# e
np.e
# cos
np.cos()

# --- Commonly used functions for numpy --- #
# real part  
.real
# imag part  
.imag
# Input a complex number, returns an absolute value.
np.absolute()
# Input a complex number, returns a radians angle.
np.angle()
# Input a complex number, returns a degrees angle.
nptoDeg(np.angle())
# Display complex in abs<deg form.
nptoAbsDeg()
# Input a complex number in Euler form(input angle in degrees):
*np.e**(nptoRad( )*1j)
# Solve the conjugate of a complex
np.conjugate()
# Power, can replace np.root(a) with a**(1/2)
a ** b


# # Fetch output
# file = open("out.txt", "w")
# file.write(str())
# file.close()


