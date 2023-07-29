import matplotlib.pyplot as plt
from PIL import Image
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
import numpy as np
from qiskit.circuit import Parameter
import math
pi = math.pi
# Load the grayscale image
image_path = "Lena.bmp"
image = Image.open(image_path)
image = image.convert('L')

# pixels changing
new_size = (64, 64)
resized_image = image.resize(new_size)
# save the origin pics
resized_image.save("Lena_origin.png")
# Convert the image to a NumPy array
pixel_array = np.array(resized_image)
#reshape
a = pixel_array.reshape((64, 64))


# change to list
e=a.tolist()

# unnesting
flattened_e = [float(item) for sublist in e for item in sublist]

# Normalized
sum_of_squares = sum(num**2 for num in flattened_e)
normalized_list = [num / math.sqrt(sum_of_squares) for num in flattened_e]

# # Calculate the sum of the squares using a loop
# sum_of_squares = 0
# for num in normalized_list:
#     sum_of_squares += num**2
# print(sum_of_squares)

# Encoding
qc = QuantumCircuit(12)
qc.initialize(normalized_list, [0,1,2,3,4,5,6,7,8,9,10,11])

# parameters added
phi = Parameter('phi')
qc.ry(phi, 0)
qc.ry(phi, 1)
qc.ry(phi, 2)
qc.ry(phi, 3)
qc.ry(phi, 4)
qc.ry(phi, 5)
qc.ry(phi, 6)
qc.ry(phi, 7)
qc.ry(phi, 8)
qc.ry(phi, 9)
qc.ry(phi, 10)
qc.ry(phi, 11)

# qc.ry(0, 0)
# qc.ry(phi, 1)
# qc.ry(phi/2, 2)
# qc.ry(phi/3, 3)
# qc.ry(phi/4, 4)
# qc.ry(phi/5, 5)
# qc.ry(phi/6, 6)
# qc.ry(phi/7, 7)
# qc.ry(phi/8, 8)
# qc.ry(phi/9, 9)
# qc.ry(phi/10, 10)
# qc.ry(0, 11)

# Encryption circuit
bc = qc.bind_parameters({phi: 256*pi*0.1})
bc.draw()
plt.show()
bc.measure([],[])

# Use Aer's AerSimulator
backend = Aer.get_backend('statevector_simulator')
job = backend.run(bc)
result = job.result()
outputstate = result.get_statevector(bc, decimals=3)
c=np.array(outputstate)
print(c)

#show vector
# counts  = result.get_counts(bc)
# plot_histogram(counts,figsize=(40,40))
# plt.show()

# Convert to real part
real_array = np.real(c)
print(real_array.shape)
#reshape
r = real_array.reshape((64, 64))

#absolute change
absolute_r = np.abs(r)
print(absolute_r)

# scale
absolute_r = absolute_r*300
print(np.max(absolute_r))

# image showing
# plt.imshow(absolute_r, cmap='gray')
# plt.axis('off')
# plt.show()

# saving
image = Image.fromarray(absolute_r.astype('uint8'))
image.save("LenaENCRY.png")
