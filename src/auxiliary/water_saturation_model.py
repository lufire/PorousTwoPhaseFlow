# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:35:42 2021

@author: lukas

according to: 
Ferreira, Rui B, et al. “A One-Dimensional and Two-Phase Flow Model of a
Proton Exchange Membrane Fuel Cell: A 1-D and Two-Phase Flow Model of a PEM
Fuel Cell.” Journal of Chemical Technology & Biotechnology 90, no. 9
(September 2015): 1547–51. https://doi.org/10.1002/jctb.4651.

"""

import sympy

# s = sympy.symbols('s')

# f_s_hi = sympy.integrate(s ** 3 * (1.47 * (1.0 - s) - 2.12 * (1.0 - s) ** 2
#                          + 1.263 * (1.0 - s) ** 3), s)
# print(f_s_hi)
# f_s_ho = sympy.integrate(s ** 3 * (1.47 * s - 2.12 * s ** 2 + 1.263 * s ** 3),
#                          s)
# print(f_s_ho)

# def func_s_hi(s):
#     return -0.180428571428571*s**7 + 0.278166666666667*s**6 \
#            - 0.2038*s**5 + 0.15325*s**4
           
# def func_s_ho(s):
#     return 0.180428571428571*s**7 - 0.353333333333333*s**6 + 0.294*s**5


# def func_s_(s, theta_deg):
#     if theta_deg < 90.0:
#         return func_s_hi(s)
#     else:
#         return func_s_ho(s)

x, y, z = sympy.symbols('x y z')
f = sympy.symbols('f', cls=sympy.Function)

def leverett_hi(s):
    return 1.47 * (1.0 - s) - 2.12 * (1.0 - s) ** 2 + 1.263 * (1.0 - s) ** 3

def leverett_ho(s):
    return 1.47 * s - 2.12 * s ** 2 + 1.263 * s ** 3

def leverett_j(s, theta):
    if theta < 90.0:
        return leverett_hi(s)
    else:
        return leverett_ho(s)

diffeq = sympy.Eq(f(x).diff(x) * f(x) ** 3 * leverett_hi(f(x)), 0)

print(sympy.dsolve(diffeq, f(x)))
print(f(x))