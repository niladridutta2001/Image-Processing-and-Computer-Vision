import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def func1(y,x,z):
    
    f1=np.exp(-(y/x)+0.5*(z/x)**2) * norm.cdf((y/z)-(z/x))
    f2= np.exp((y/x)+0.5*(z/x)**2) * norm.cdf(-(y/z)-(z/x))
    return (f2+f1)/(2*x)

def func2(y,x,z):
    a=np.sqrt(2*np.pi)
    f3=np.exp(-(y/x)+0.5*(z/x)**2)*(((z/a)*np.exp(-0.5*((y/z)-(z/x))**2))+((y-((z**2)/x))*norm.cdf((y/z)-(z/x))))
    f4=np.exp((y/x)+0.5*(z/x)**2)*((-1*(z/a)*np.exp(-0.5*((y/z)+(z/x))**2))+((y+((z**2)/x))*norm.cdf(-(y/z)-(z/x))))
    #  f3=func1(x,y,z)*(2*x)
    return (f3+f4)/(2*x)
def func3(y,x):
    return np.exp(-abs(y)/x)/(2*x)
def func4(y,z):
    a=np.sqrt(2*np.pi)
    return np.exp(-0.5*(y/z)**2)/(a*z)
if __name__=="__main__":
    x_values = np.linspace(-1.5,1.5, 1000)
    sigmax=1
    sigmaz=np.sqrt(0.1)
    y_values1 = func1(x_values, sigmax, sigmaz)
    y_values3 = func3(x_values, sigmax)
    y_values4 = func4(x_values, sigmaz)
    y_values2 = func2(x_values, sigmax, sigmaz)/func1(x_values, sigmax, sigmaz)
    area = np.trapz(y_values1, x_values)
    print(area)


    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot the custom function in the first subplot
    axes[0].plot(x_values, y_values1,label='PDF of y')
    axes[0].plot(x_values, y_values3, linestyle='--', color='red', label='Prior on X')
    axes[0].plot(x_values, y_values4, linestyle='-', color='orange', label='PDF of Z(noise)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('PDF plots')
    axes[0].legend()
    axes[0].grid(True)

    # Plot the sine function in the second subplot
    axes[1].plot(x_values, y_values2,label='Estimate of X')
    axes[1].plot(x_values, x_values,  color='orange', label='Line: y = x')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('x')
    axes[1].set_title('Estimate for x')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()