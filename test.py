import matplotlib.pyplot as plt
from solidspy import preprocesor

mesh = preprocesor.rect_grid(1.0,1.0,100,100)
print(mesh)
print(len(mesh))
print(type(mesh[0]))
plt.figure()
plt.plot(mesh[0],mesh[1])
plt.figure()