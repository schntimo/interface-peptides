import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trp


# %% Inputs
filepath = './test_colvar_out.pmf'
Range_Ref = np.array([65,70])
T = 300

R = 8.314
kcaltoJ = 4184
Na = 6.022025e23

# %% Code

# Read data from the file into NumPy arrays
data = np.genfromtxt(filepath, skip_header=1)
x_values = data[:, 0]
y_values = data[:, 1]

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='.', linestyle='-', color='b', label='Data')
plt.xlabel('Distance [A]')
plt.ylabel('PMF [kcal/mol]')
plt.title('Data Plot')
plt.grid(True)
plt.legend()
plt.show()



Max_integrate = Range_Ref[0]
PMF = y_values
x_coord = x_values
PMF_ref = np.mean(PMF[(x_coord > Range_Ref[0]) & (x_coord < Range_Ref[1])])
PMF_norm = (PMF-PMF_ref)*kcaltoJ/(T*R)

# fig, ax = plt.subplots()
# plt.plot(x_coord, PMF_norm, 'k-')
# plt.xlabel("Distance [nm]")
# plt.ylabel("PMF [$k_B T$ ]")
# plt.title("POTENTIAL OF MEAN FORCE")
# plt.plot([0, 70], [0, 0],"k--", lw=0.7)
# # plt.xlim([0, 3])
# # plt.ylim([-1.8, 3])
# plt.show()

Mayer_f = np.exp(-PMF_norm) - 1
x_coord_ext = np.append(0, x_coord)
Mayer_f = np.append(-1, Mayer_f)

# fig, ax = plt.subplots()
# plt.plot(x_coord_ext, Mayer_f, 'k-')
# plt.xlabel("Distance [nm]")
# plt.ylabel("Mayer f-function [-]")
# plt.title("MAYER F-FUNCTION")
# plt.plot([0, 70], [0, 0],"k--", lw=0.7)
# # plt.xlim([0, 3])
# # plt.ylim([-1.5, 1])
# plt.show()

Integral = Mayer_f*x_coord_ext**2

# fig, ax = plt.subplots()
# plt.plot(x_coord_ext, Integral, 'k-')
# plt.xlabel("Distance [nm]")
# plt.ylabel("Integral function [nm$^2$[}]")
# plt.title("INTEGRAL FUNCTION")
# plt.plot([0, 70], [0, 0],"k--", lw=0.7)
# # plt.xlim([0, 3])
# # plt.ylim([-1, 1])
# plt.show()

Integral_calc = Integral[x_coord_ext < Max_integrate]
x_coord_ext_calc = x_coord_ext[x_coord_ext < Max_integrate]

Integral_value = trp(Integral_calc, x_coord_ext_calc)
B = -2*np.pi*Integral_value/1000

print(f"SECOND VIRIAL COEFFICIENT = {B} nm3 \n")
