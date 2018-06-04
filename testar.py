import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*np.pi*t)


fig = plt.figure()
plt.plot(t,s)
#plt.show()
plt.draw()
plt.pause(1)
k = input()
print(k)
plt.close(fig)

		'''
		R_aux = np.zeros((kernel_size, kernel_size))
		G_aux = np.zeros((kernel_size, kernel_size))
		B_aux = np.zeros((kernel_size, kernel_size))
		
		if(mode == 0):
			R_aux = R[l:l+kernel_size, c:c+kernel_size]
		
		elif(mode == 1):
			G_aux = G[l:l+kernel_size, c:c+kernel_size]
	
		elif(mode == 2):
			B_aux = B[l:l+kernel_size, c:c+kernel_size]
			
		else:
			R_aux = R[l:l+kernel_size, c:c+kernel_size]
			G_aux = R[l:l+kernel_size, c:c+kernel_size]
			B_aux = R[l:l+kernel_size, c:c+kernel_size]
		'''	