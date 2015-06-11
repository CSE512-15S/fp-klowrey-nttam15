import sabotload as sb
import glob
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import json
#import pylab as Plot

import tsne

A00 = sorted(glob.glob('/mnt/traj/*_A00pol.mat'))
A01 = sorted(glob.glob('/mnt/traj/*_A01pol.mat'))
A02 = sorted(glob.glob('/mnt/traj/*_A02pol.mat'))
A03 = sorted(glob.glob('/mnt/traj/*_A03pol.mat'))
#hidden = sorted(glob.glob('/mnt/traj/T00088S00004*.mat'))


first = True
i = 0
tsne_data = []
for f0, f1, f2, f3 in zip(A00, A01, A02, A03):
    a00 = sb.sabotload(f0)

    a01 = sb.sabotload(f1)
    a02 = sb.sabotload(f2)
    a03 = sb.sabotload(f3)
    
    layers = np.hstack((a00.T[0:1][:],a01.T[0:1][:],a02.T[0:1][:],a03.T[0:1][:]))
    #layers = np.hstack((a01.T[0:1][:],a02.T[0:1][:],a03.T[0:1][:]))
    i += 1
    if first :
        l1 = np.array(layers) 
        first = False
    else :
        l1 = np.vstack((l1, layers))
    #print i, np.shape(a00), np.shape(a01), np.shape(a02), np.shape(a03), np.shape(layers)
    #print f0, f1, f2, f3 
    #tsne_data.append(a[0][0])

r,c = np.shape(l1)
print "Data Loaded:", np.shape(l1)

no_dims = 1
initial_dims = r
perplexity = 30

Y = tsne.tsne(l1.T, no_dims, initial_dims, perplexity, 500)

no_tsne_inputs = False
if no_tsne_inputs:
    inputs=np.linspace(0,1,75)
    print np.shape(inputs), np.shape(Y[:,0])
    Y = np.hstack((inputs, Y[:,0]))
    
    output=np.linspace(0,1,20)
    print np.shape(Y), np.shape(output) 
    Y = np.hstack((Y, output))
else:
    output=np.linspace(0,1,20)
    Y = np.hstack((Y[:,0], output))
    Y = np.hstack((Y, output))
print np.shape(Y.T)

out = {"points":Y.T.tolist()}

#outstring = "{ \"points\":"+str(Y.T)+"}"
outstring = json.JSONEncoder().encode(out)
fid = open('robot/biped_all_points.json','w')
fid.write(outstring)
fid.close()



