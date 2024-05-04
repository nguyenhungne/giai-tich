import sympy as sp
from sympy import symbols
from sympy import diff, sin, exp, plot 
import math as math
import numpy as np
import matplotlib.pyplot as plt

x = sp.symbols('x')

A = int(input())

def ex1a(A):
    x = sp.symbols('x')
    Fx = x**2 - 2*A*x - A**2
    Gx = -x**2 + 4*A*x + A**3

    solutions = sp.solve(Fx-Gx, x)
    x1 = float(solutions[0])
    y1 = float(Fx.subs(x, x1))
    x2 = float(solutions[1])
    y2 = float(Fx.subs(x, x2))

    print("1a. Intersection point 1 : ({0};{1})".format(x1, y1))
    print("1a. Intersection point 2 : ({0};{1})".format(x2, y2))
    x = 0
    if A >= 10 and A <= 40:
        x = np.linspace(-1000, 1000)
    else :
        x = np.linspace(-2000, 2000)
    Fx = x**2 - 2*A*x - A**2
    Gx = -x**2 + 4*A*x + A**3
    plt.plot(x, Fx, label='f(x)', color='blue')
    plt.plot(x, Gx, label='g(x)', color='red')
    plt.plot(x1, y1, 'o',  color='green')
    plt.plot(x2, y2, 'o',  color='green')
    plt.title('Question 1a')

    plt.show()
    


def ex1b(A):
    x = sp.symbols('x')
    Fx = x**2 - 2*A*x - A**2
    derivativeFx = diff(Fx, x)
    x0 = 0
    y0 = -A**2
    slope = derivativeFx.subs(x, x0)
    tangentLine = slope*(x - x0) + y0
    x3 = float(sp.solve(tangentLine, x)[0])
    y3 = float(Fx.subs(x, x3))
    print("Equation of the tangent line to the curve f(x) : {0}".format(tangentLine))
    shiftFx = Fx - 4*A**3
    solutions = sp.solve(shiftFx - tangentLine, x)
    x1 = float(solutions[0])
    y1 = float(shiftFx.subs(x, x1))
    x2 = float(solutions[1])
    y2 = float(shiftFx.subs(x, x2))
    print("1b. Intersection point 1 : ({0};{1})".format(x1, y1))
    print("1b. Intersection point 2 : ({0};{1})".format(x2, y2))

    x = 0
    if A >= 10 and A <= 40:
        x = np.linspace(-1000, 1000)
    else :
        x = np.linspace(-2000, 2000)
    Fx = x**2 - 2*A*x - A**2
    tangentLine = slope*(x - x0) + y0
    shiftFx = Fx - 4*A**3
    plt.plot(x, Fx, label='f(x)', color='blue')
    plt.plot(x3, y3, 'o', markersize=4, color='green')
    plt.plot(x, tangentLine, label='tangent line to f(x)', color='yellow')
    plt.plot(x, shiftFx, label='shift f(x)', color='red')
    plt.plot(x1, y1, 'o',  color='black')
    plt.plot(x2, y2, 'o',  color='black')
    plt.title('Question 1b')
    plt.show()


def ex1c(A):
    x = sp.symbols('x')
    Fx = x**2 - 2*A*x - A**2
    derivativeFx = sp.diff(Fx, x)

    x0 = 0
    y0 = -4*A**3


    tangentPoints = sp.solve(derivativeFx - (Fx - y0) / (x - x0), x)

    x_vals = 0
    if A >= 10 and A <= 40:
        x_vals = np.linspace(-1000, 1000)
    else :
        x_vals = np.linspace(-2000, 2000)
    Fx_lambda = sp.lambdify(x, Fx, "numpy")

    plt.plot(x_vals, Fx_lambda(x_vals), label='f(x)', color='blue')
    plt.plot(x0, y0, 'o', markersize=4, color='green')

    yTangentPoint1 = Fx.subs(x, tangentPoints[0])
    tangentLine1 = float(derivativeFx.subs(x, tangentPoints[0]))*(x - float(tangentPoints[0])) + float(yTangentPoint1)
    print("Equation of the tangent line 1 to the curve f(x) : {0}".format(tangentLine1))
    yTangentPoint2 = Fx.subs(x, tangentPoints[1])
    tangentLine2 = float(derivativeFx.subs(x, tangentPoints[1]))*(x - float(tangentPoints[1])) + float(yTangentPoint2)
    print("Equation of the tangent line 2 to the curve f(x) : {0}".format(tangentLine2))

    plt.plot(tangentPoints[0], yTangentPoint1, 'o', markersize=4, color='black')
    plt.plot(tangentPoints[1], yTangentPoint2, 'o', markersize=4, color='black')
    tangentLine1_lambda = sp.lambdify(x, tangentLine1, "numpy")
    tangentLine2_lambda = sp.lambdify(x, tangentLine2, "numpy")
    plt.plot(x_vals, tangentLine1_lambda(x_vals), label='tangent line 1', color='skyblue')
    plt.plot(x_vals, tangentLine2_lambda(x_vals), label='tangent line 2', color='orange')




    plt.title('Question 1c')
    plt.legend()
    plt.show()
ex1a(A)
ex1b(A)
ex1c(A)