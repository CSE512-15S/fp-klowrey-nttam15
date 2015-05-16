/* global labels */
var tsnejs = require('./scripts/tsne');
var convnetjs = require('./scripts/convnet-min');

var opt = {epsilon: 10, perplexity: 30, dim: 3}; // epsilon is learning rate (10 = default)
var tsne = new tsnejs.tSNE(opt); // create a tSNE instance
var convnet = new tsnejs.tSNE(opt); // create a tSNE instance



// make layers
var layer_defs = [];
// input layer of size 1x1x2 (all volumes are 3D)
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// some fully connected layers
layer_defs.push({type:'fc', num_neurons:3, activation:'sigmoid'});
//layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
// a softmax classifier predicting probabilities for two classes: 0,1
layer_defs.push({type:'softmax', num_classes:2});
//layer_defs.push({type:'regression', num_neurons:1});

// make network from layers above
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
save_net_to_json(net, 'untrained_network.json');

//var trainer = new convnetjs.SGDTrainer(net, 
//              {learning_rate:0.2, momentum:0.0, batch_size:10, l2_decay:0.001});
var trainer = new convnetjs.Trainer(net, 
              {method: 'adadelta', batch_size:10, l2_decay:0.001});
 
// train on this datapoint, saying [0.5, -1.3] should map to value 0.7:
// note that in this case we are passing it a list, because in general
// we may want to  regress multiple outputs and in this special case we 
// used num_neurons:1 for the regression to only regress one.
//var x = new convnetjs.Vol([0.5, -1.3]);

// forward prop the data
var start = new Date().getTime();

// xor
data = [];
data.push([0.0, 0.0]);
data.push([0.0, 1.0]);
data.push([1.0, 0.0]);
data.push([1.0, 1.0]);
// classfication needs class number,

labels = [];
labels.push(0);
labels.push(1);
labels.push(1);
labels.push(0);
// regression needs list of values
//labels = [];
//labels.push([0]);
//labels.push([1]);
//labels.push([1]);
//labels.push([0]);
var N = data.length;

var x = new convnetjs.Vol(1,1,2);

function train_network(d, l) {
   var start = new Date().getTime();
   // 1 x 1, with a depth of 2 ( vector length 2 )
   //x.w = d[ix];
   var avloss = 0.0;
   for(var iters=0;iters<2000;iters++) {
      for(var ix=0;ix<N;ix++) {
         x.w = d[ix];
         var stats = trainer.train(x, l[ix]);
         avloss += stats.loss;
      }
   }
   avloss /= N*iters;

   var end = new Date().getTime();
   var time = end - start;

   console.log('loss = ' + avloss + ', 100 cycles through data in ' + time + 'ms');
}

train_network(data, labels);

for(var ix=0;ix<N;ix++) {
   x.w = data[ix];
   var predicted_values = net.forward(x);
   console.log('in: ' + data[ix]+' goal: '+labels[ix]+' out: '+predicted_values.w[0]+' '+predicted_values.w[1]);
}


function layer_data(network) {
   var num_layers = network.layers.length;
   for (var nl=0; nl<num_layers; nl++) {
      network.layers[nl].out_act.w[0];
   }
}

save_net_to_json(net, 'trained_network.json');


function save_net_to_json(network, filename) {
   // save convnetjs to file as json
   var json = network.toJSON();
   var str = JSON.stringify(json, null, 3);
   var fs = require('fs');
   fs.writeFile(filename, str, function(err) {
      if (err) {
         console.log(err);
      } else {
         console.log("json saved to "+filename);
      }
   });
}

// get coactivations with test dataset test_d
function get_coactivation_data(network, test_d) {
   var N = test_d.length;
   for(var ix=0;ix<N;ix++) {
      x.w = test_d[ix];
      var predicted_values = network.forward(x);
      //console.log('in: ' + data[ix]+' goal: '+labels[ix]+' out: '+predicted_values.w[0]+' '+predicted_values.w[1]);
   }
}



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
/*
   var dists = [
   [1.0, 0.1, 0.2, 0.6, 0.9, 1.02, 10.2, 100],
   [0.1, 1.0, 0.3, 0.6, 0.9, 1.02, 10.2, 100],
   [0.2, 0.1, 1.0, 0.6, 0.9, 1.02, 10.2, 100]];
   tsne.initDataDist(dists);

   for(var k = 0; k < 500; k++) {
   tsne.step(); // every time you call this, solution gets better
   }

   var Y = tsne.getSolution(); // Y is an array of 2-D points that you can plot

   console.log("Dists:");
   console.log(dists);
   console.log("Output:");
   console.log(Y)
   */

