/* global labels */
var convnetjs = require('./scripts/convnet-min');

// make network from file
var net = new convnetjs.Net();
var mnist_file = require('./simple_mnist_net.json');
net.fromJSON(mnist_file);

//var trainer = new convnetjs.SGDTrainer(net, 
//              {learning_rate:0.2, momentum:0.0, batch_size:10, l2_decay:0.001});
var trainer = new convnetjs.Trainer(net, 
              {method: 'adadelta', batch_size:10, l2_decay:0.001});
 
// forward prop the data

// FORMAT for tsne:
// vectors of values
//
// for neural networks, that means activations for the list of inputs
//
// input 1, neuron 1, neuron 2, neuron 3 ... neuron n
// input 2, neuron 1, neuron 2, neuron 3 ... neuron n
// input 3, neuron 1, neuron 2, neuron 3 ... neuron n

// TSNE stuff
// initialize data. Here we have 3 points and some example pairwise dissimilarities
//var from_file = require('./matlab/simple_mnist.json'); //tsne data from matlab
var helperjs = require('./helper.js');
var helper = new helperjs.helper();
var from_file = require('./matlab/simple_mnist_v2.json'); //tsne data from matlab
var points = from_file.points
console.log("Points: "+points.length);

// NORMALIZE the tsne outputs
//console.log(points);
console.log("Normalizing by layers");
links = helper.normalizeBetweenLayers(net, points);
n_points = helper.normalizeByLayers(net, points, 0, 7);


//for (var l=0; l<(layers-1); l++) {
//   var L = net.layers[l];
//   console.log("Layer "+l+": "+L.layer_type);
//}
////////// write out json file of data for sankey

var net_opt ={'name':'no_conv_mnist', 'node_scale':10, 'link_scale':4, 'link_opcty':8,
    'nodeWidth':40,
    'nodePadding':4};
var collapse_input = 1;
var slim_thresh = 0.7;
helper.dataToSankey(net, n_points, links, net_opt, 'data/mnist_sankey_v2.json', collapse_input, slim_thresh);



