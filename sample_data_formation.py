import numpy as np
import math

#SAMPLE DATA FORMER
#the y axis is the direction when robot opens its arms wide. the notation may be confusing here since swapped

z = [1] * 200

#final y = 1.1, x = 0, z = 1


# DATA FOR CIRCLE TRAJECTORY

# Generating an array of 200 samples between 0 and 1.1
y = np.linspace(-1.1, 0, 200) 

multiplication = [k * k for k in y]
subtraction  = [c - d for c, d in zip([1.1*1.1]*200, multiplication)]
x = [math.sqrt(r) for r in subtraction]


trajectory = list(zip(x,y,z))

np.save('circle_trajectory',trajectory)




# DATA FOR STEP TRAJECTORY

y = np.linspace(-1.1, 0, 200) 


x = [0]* 120 + np.linspace(0,1.1,30).tolist() +  [1.1]*20
trajectory = list(zip(x,y,z))

np.save('one_step_trajectory',trajectory)



# DATA FOR LINEAR TRAJECTORY

y = np.linspace(-1.1, 0, 200) 

x = - 1*y * 0.5 + [1.1]*200 
trajectory = list(zip(x,y,z))

np.save('linear_trajectory',trajectory)


# DATA FOR SINUS TRAJECTORY

y = np.linspace(-1.1, 0, 200) 

# Computing y as a function of x
#from 0 to 0.6

#0.5*sin(2*pi*y)
prepend = [0.5]*70
append = [(0.1 * k + 0.5) for k in [math.sin(2 * math.pi * 4 * a) for a in y[70:-30]]]
final = [1.1]*30
m = prepend + append
trajectory = list(zip(m,y,z))

np.save('sinus_trajectory',trajectory)




#BEWARE: circle requires smaller t, sinus-like ones require larger t. others not sure but don't variate much. TODO: DISCUSS