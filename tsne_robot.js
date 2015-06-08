/* global labels */
var convnetjs = require('./scripts/convnet-min');

// make network from file
var net = new convnetjs.Net();

// make layers
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:76});
layer_defs.push({type:'fc', num_neurons:250, activation:'relu'});
layer_defs.push({type:'fc', num_neurons:250, activation:'relu'});
layer_defs.push({type:'fc', num_neurons:250, activation:'relu'});
var output_type = 'regression'
if (output_type == 'softmax') {
    layer_defs.push({type:'softmax', num_classes:2});
}
else {
    layer_defs.push({type:'regression', num_neurons:20});
}

net.makeLayers(layer_defs);

var trainer = new convnetjs.Trainer(net, {method: 'adadelta', batch_size:10, l2_decay:0.001});

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
var from_file = require('./robot/humanoid_points.json'); //tsne data from python 
//var from_file = require('./robot/biped_all_points.json'); //tsne data from python 
var points = from_file.points
console.log("Points: "+points.length);
console.log(points[points.length-20]);

// NORMALIZE the tsne outputs
//console.log(points);
console.log("Normalizing by layers");
links = helper.normalizeBetweenLayers(net, points, 0, 8);
console.log("LINK LENGTH: "+links.length);
n_points = helper.normalizeByLayers(net, points, 0, 8);


//for (var l=0; l<(layers-1); l++) {
//   var L = net.layers[l];
//   console.log("Layer "+l+": "+L.layer_type);
//}
////////// write out json file of data for sankey

var net_opt ={'name':'humanoid',
    'node_scale':10,
    'link_scale':4,
    'link_opcty':64,
    'nodeWidth':40,
    'nodePadding':0};

var collapse_input = -1;
var slim_thresh = 0.95;
helper.dataToSankey(net, n_points, links, net_opt, 'data/humanoid.json', collapse_input, slim_thresh);

//var slim_thresh = 0.8;
//helper.dataToSankey(net, n_points, links, net_opt, 'data/biped_all.json', collapse_input, slim_thresh);

