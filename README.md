# Neural Network Visualization

Neural networks are frequently treated as black box function approximators.


Current status:

a non-convolutional neural network for mnist was trained and put into index.html (two versions).
matlab was used for tsne -- pain to get activations data out of the mnist.html run-time...

when using karpathy's stuff, we're ignoring conv and fc layers (just getting post-synaptic activations)
tsne for the mnist networks only works when excluding the input and output layers from tSNE -- the weights for these are going to be odd. they are added back in with random placements to try to have 'even' distribution in the render'
doesn't quite make good links to the output layer; need to modify sankey.js to specifically include the last layer, or modify tsne_mnist.js

todo: 

switching between the different networks (so var xor and mnist)
online training and rendering of a network (xor?)
