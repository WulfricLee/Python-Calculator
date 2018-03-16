#!/usr/bin/python

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
# Sientific expression
#2e-2 = 0.02
#2e-3 = 0.002

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

# === Quaternion ===
# 
# Quternion multiplication function
# q1 and q2 are two quaternions, quaternion multiplication is NOT communicative.
def quaternion_multiplication(q1, q2):
	return [ \
		q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3], \
		q1[1] * q2[0] + q1[0] * q2[1] - q1[3] * q2[2] + q1[2] * q2[3], \
		q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2] - q1[1] * q2[3], \
		q1[3] * q2[0] - q1[2] * q2[1] + q1[1] * q2[2] + q1[0] * q2[3]]
# 
# Get the rotation in quaternion form. Where theta is rotate angle(in radian and follows right-hand rule),
# and axis is the rotation axis in quaternion form, this axis must be a unit vector.
def get_rotation_in_quaternion_form(theta, axis):
	cos_part = cos(theta/2)
	sin_part = sin(theta/2)
	return [cos_part, sin_part * axis[1], sin_part * axis[2], sin_part * axis[3]]
# 
# Vector rotation in quaternion form, returns the vector after rotation in quaternion form.
# q stands for the rotation in quaternion form, 
# v stands for the original vector before rotation in quaternion form.
def get_vector_after_rotation(q, v):
	q_conj = [q[0], -q[1], -q[2], -q[3]]
	return quaternion_multiplication(quaternion_multiplication(q, v), q_conj)
# 
# phi -> x axis, theta -> y axis, psi -> z axis.
# The Euler angles follow ZYX order.
def eular_to_quaternion(phi, theta, psi):
	return [ \
	cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2), \
	sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2), \
	cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2), \
	cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2)]
# 
def eular_to_quaternion_in_decimal(phi, theta, psi):
	return [ \
	(cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2)).evalf(), \
	(sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2)).evalf(), \
	(cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2)).evalf(), \
	(cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2)).evalf()]
# 
def quaternion_to_eular(q):
	return [ \
	atan2(2 * (q[0] * q[1] + q[2] * q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2), \
	asin(2 * (q[0] * q[2] - q[3] * q[1])), \
	atan2(2 * (q[0] * q[3] + q[1] * q[2]), q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2)]
# 
def quaternion_to_eular_in_degree(q):
	return [ \
	toDeg(atan2(2 * (q[0] * q[1] + q[2] * q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)), \
	toDeg(asin(2 * (q[0] * q[2] - q[3] * q[1]))), \
	toDeg(atan2(2 * (q[0] * q[3] + q[1] * q[2]), q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2))]
# ==================


############################

# ===== To use NumPy ===== #
import numpy as np

# Set linewidth display 
# https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.set_printoptions.html
np.set_printoptions(linewidth = 200)

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


