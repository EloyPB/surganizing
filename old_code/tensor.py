import numpy as np


a = np.array([1, 0, 1, 0])
b = np.array([[[1,2,3],[-1,-2,-3]],[[4,5,6],[-4,-5,-6]],[[7,8,9],[-7,-8,-9]],[[10,11,12],[-10,-11,-12]]])
c = np.tensordot(a, b, axes=1)

print(a)
print(b)
print(c)

print()
print("masked inner product")
mask = np.array([0,0,0,1])
ma = np.ma.masked_array(a, mask)
mb = np.array([[[1,10,100,1000],[2,20,200,2000],[3,30,300,3000]],[[-1,-1,-1,-1],[-2,-2,-2,-2],[-3,-3,-3,-3]]])
mc = np.ma.inner(ma, mb)

print(ma)
print(mb)
print(mb.shape)
print(mc)

print("weight")
w = np.ma.inner(mc[:,:,np.newaxis], ma[:,np.newaxis])
print(w)
print('shape w: ', w.shape)

