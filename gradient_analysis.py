import numpy as np




def fit_line(x, y):
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    return slope, intercept

tf_liv = [12.56, 5.45, 6.96]
tf_moss = [8.76, 1.45, 0.38]


tf_t = [1,2,3]

c_lif = [19.54, 12.73]
c_moss = [0.81, 0.08]


c_t = [1,2]

slope, intercept = fit_line(tf_liv, tf_t)
print("Slope TF:", slope)
print("Intercept TF:", intercept)

slope, intercept = fit_line(c_lif, c_t)
print("Slope C:", slope)
print("Intercept C:", intercept)

slope, intercept = fit_line(tf_moss, tf_t)
print("Slope TF Moss:", slope)
print("Intercept TF Moss:", intercept)

slope, intercept = fit_line(c_moss, c_t)
print("Slope C Moss:", slope)
print("Intercept C Moss:", intercept)