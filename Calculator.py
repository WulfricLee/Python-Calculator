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
	x = x/pi * 180
	return x
def toRad(x):
	x = x/180 * pi
	return x

# from mpmath import *
# mp.dps = 15; mp.pretty = True

a,b,c,d,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,F,G,H,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z = symbols('a b c d f g h i j k l m n o p q r s t u v w x y z A B C D F G H J K L M N O P Q R S T U V W X Y Z')
x = symbols('x')
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
# Simplify a expression
simplify(expression)
# Factoring a expression
factor(expression)
# Expand a expression
expand(expression)
# Sientific expression
#2e-2 = 0.02
#2e-3 = 0.002


############################

# ===== To use NumPy ===== #
import numpy as np
# Deg-Rad transform functions
def toDeg(x):
	x = x/np.pi * 180
	return x
def toRad(x):
	x = x/180 * np.pi
	return x
# Display complex in abs<deg form.
def printAbsDeg(x):
	abs = np.absolute(x)
	deg = toDeg(np.angle(x))
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
# Input a complex number, returns a absolute value.
np.absolute()
# Input a complex number, returns a radians angle.
np.angle()
# Input a complex number, returns a degrees angle.
toDeg(np.angle())
# Display complex in abs<deg form.
toAbsDeg()
# Input a complex number in Euler form(input angle in degrees):
*np.e**(toRad( )*1j)
# Solve the conjugate of a complex
np.conjugate()
# Power, can replace np.root() with a**(1/2)
a ** b


# # Fetch output
# file = open("out.txt", "w")
# file.write(str())
# file.close()


