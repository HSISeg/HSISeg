import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, cuda

x = np.arange(5 * 3 * 10 * 10, dtype='f').reshape(5, 3, 10, 10)

l = L.Convolution2D(3, 4, 5)# (input_channel, nfilter, patch_size-p)

y = l(x)

y1 = F.relu(y) #(5,4,6,6)

# y2 = F.reshape(y1,(5, 2, 2, 6*6)) # (no_of_inputs, no of bands, band size, , nfilter, patch_size-p)

a1, a2 = F.split_axis(y1,2,axis=1)

# a3 = F.reshape(a1,(a1.shape[0],a1.shape[1]*a1.shape[2],a1.shape[3]))

a4 = F.swapaxes(a1, axis1=1,axis2=3)

a5 = F.flip(a4,3) #(5,6,6,2)

l2 = L.Convolution2D(6, 20, (6,1)) #(image_length- x, nfilter, patch_size(image_width- y,p))

a6 = l2(a5) #(5,20,1,2)

a7 = F.relu(a6)

l3 = L.Convolution2D(20,20,(1,1))

a8 = l3(a7) #(5, 20, 1, 2)

a9 = F.relu(a8)

l4 = L.Convolution2D(20,10,(1,1))

a10 = l4(a9) #(5, 10, 1, 2)

a11 = F.relu(a10)

l5 = L.Convolution2D(10,5,(1,2))

a12 = l5(a11) #(5, 5, 1, 1)

a13 = F.relu(a12)

a14 = F.reshape(a13,(a13.shape[0],a13.shape[1]*a13.shape[3],a12.shape[2]))  #(5, 5, 1)

a15 = F.dstack((a14,a14,a14))

a16 = F.reshape(a15,(a15.shape[0],a15.shape[1]*a15.shape[2]))

l6 = L.Linear(a16.shape[1],100)

a17 = l6(a16)

a18 = F.relu(a17)

a19 = F.dropout(a18)

l7 = L.Linear(100,1)

a20 = l7(a19)

xxx = np.arange(55 * 44 * 22 * 22, dtype='f').reshape(55, 44, 22, 22)


xxx = np.arange(55 * 44 * 22 * 22, dtype='f').reshape(55, 44, 22 * 22)
rotxxx = np.rot90(xxx,-1,(1,2))
y = F.swapaxes(xxx, axis1=1,axis2=2)
yyy= F.flip(y,2)
np.equal(rotxxx,yyy.data).all()
xxx = np.arange(2 * 3 * 2 * 2, dtype='f').reshape(2, 3, 2, 2)