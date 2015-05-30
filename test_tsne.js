/* global labels */
var tsnejs = require('./scripts/tsne');
var convnetjs = require('./scripts/convnet-min');

// make layers
var layer_defs = [];
// input layer of size 1x1x2 (all volumes are 3D)
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:3});
// some fully connected layers
layer_defs.push({type:'fc', num_neurons:4, activation:'sigmoid'});
//layer_defs.push({type:'fc', num_neurons:20, activation:'relu'});
// a softmax classifier predicting probabilities for two classes: 0,1

//layer_defs.push({type:'softmax', num_classes:2});
layer_defs.push({type:'regression', num_neurons:1});

// make network from layers above
var no_train = new convnetjs.Net();
var net = new convnetjs.Net();
no_train.makeLayers(layer_defs);
net.makeLayers(layer_defs);

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
labels = [];
data.push([0.0, 0.0, 0.0]); labels.push([0]);
data.push([0.0, 1.0, 0.0]); labels.push([1]);
data.push([1.0, 0.0, 0.0]); labels.push([1]);
data.push([1.0, 1.0, 0.0]); labels.push([1]);
data.push([0.0, 0.0, 1.0]); labels.push([1]);
data.push([0.0, 1.0, 1.0]); labels.push([1]);
data.push([1.0, 0.0, 1.0]); labels.push([1]);
data.push([1.0, 1.0, 1.0]); labels.push([0]);

// classfication needs class number,

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
   for(var iters=0;iters<4000;iters++) {
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
for(var ix=0;ix<N;ix++) {
   x.w = data[ix];
   var predicted_values = no_train.forward(x);
   console.log('in: ' + data[ix]+' goal: '+labels[ix]+' out: '+predicted_values.w[0]+' '+predicted_values.w[1]);
}
//save_net_to_json(net, 'trained_network.json');

var test_data = [];
test_data.push(data[0]);
test_data.push(data[1]);
test_data.push(data[2]);
test_data.push(data[3]);
test_data.push(data[4]);
test_data.push(data[5]);
test_data.push(data[6]);
test_data.push(data[7]);
test_data.push(data[0]);
test_data.push(data[1]);
test_data.push(data[2]);
test_data.push(data[3]);
test_data.push(data[4]);
test_data.push(data[5]);
test_data.push(data[6]);
test_data.push(data[7]);

var tsne_data = get_coactivation_data(net, test_data);
var raw_data = get_coactivation_data(no_train, test_data);

function layer_data(network, act) {
   var layers = network.layers.length;
   var count = 0;
   for (var l=0; l<(layers); l++) {
      //console.log("layer "+l);
      var neurons = network.layers[l].out_act.w.length;
      for (var nw=0; nw<neurons; nw++) {
         //console.log("\tweight "+nw);
         weight = network.layers[l].out_act.w[nw];
         act[count].push(weight);
         count = count+1;
      }
   }
}

// get coactivations with test dataset test_d
function get_coactivation_data(network, test_d) {
   var N = test_d.length;
   var layers = network.layers.length;
   var activations = [];
   var count = 0;
   for (var l=0; l<(layers); l++) {
      var neurons = network.layers[l].out_act.w.length;
      for (var nw=0; nw<(neurons); nw++) {
         activations[count] = [];
         count = count+1;
      }
   }
   console.log(count+" total neurons");

   // for each test input, net forward...
   for (var ix=0;ix<N;ix++) {
      x.w = test_d[ix];
      var predicted_values = network.forward(x);
      // then get each layer of data
      //for (var l=0; l<(layers); l++) {
      //console.log('Getting activations for layer '+l+'\'s neurons');
      layer_data(network, activations);
      //}
      //console.log('in: ' + data[ix]+' goal: '+labels[ix]+' out: '+predicted_values.w[0]+' '+predicted_values.w[1]);
   }
   //console.log('\t\t'+num_weights+' weights for '+N+' inputs');
   //console.log(activations);
   return activations;
}

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
var opt = {epsilon: 10, perplexity: 30, dim: 1};

function get_tsne(options, data, iters) {
   var tsne = new tsnejs.tSNE(options); // create a tSNE instance
   tsne.initDataDist(tsne_data);
   if (iters <= 0) {
      return null;
   }
   for(var k = 0; k < iters; k++) {
      tsne.step(); // every time you call this, solution gets better
   }
   return tsne.getSolution();
}

console.log("TSNE input:");
console.log(tsne_data.length+" by "+tsne_data[0].length +" test inputs");
console.log(tsne_data);
console.log("TSNE Output:");
var points = get_tsne(opt, tsne_data, 500);

// NORMALIZE the tsne outputs
console.log(points);
if (opt.dim == 1) {
   console.log("Normalizing by layers");
   var layers = net.layers.length;
   var x = 0;
   for (var l=0; l<(layers); l++) {
      var neurons = net.layers[l].out_act.w.length;

      var l_norm = 0;
      var min = Math.min.apply(null, points.slice(x,x+neurons));
      var div = Math.max.apply(null, points.slice(x,x+neurons))-min;

      //console.log(points.slice(x,x+neurons)+" min: "+min);
      for (var n=0; n<neurons; n++) {
         //if (Math.abs(div) > 1e-9) {
         //points[x+n][0] = (points[x+n][0]-min) / (div);
         //}
         l_norm += points[x+n][0] * points[x+n][0];
      }
      var mag = Math.sqrt(l_norm); 
      for (var n=0; n<neurons; n++) {
         var val = points[x+n][0] / mag;
         val = val + 1.0;
         points[x+n][0] = val / 2.0;
      }
      x = x+n;
   }
}
if (opt.dim == 2) {
   for (var i=0; i<points.length; i++) {
      var x =points[i][0];
      var y =points[i][1];
      var mag = Math.sqrt(x*x + y*y); 
      points[i] = [x / mag, y/mag];
   }
}

console.log(points);
console.log("TSNE untrained");
console.log(get_tsne(opt, raw_data, 500));

/*
tsne.initDataDist(tsne_data);
for(var k = 0; k < 500; k++) {
   tsne.step(); // every time you call this, solution gets better
}

*/
//data_to_rawtsne(net, "data/tsne_points.json");

function data_to_rawtsne(network, filename) {
   var data = {"network":[], "links":[]};
   var layers = network.layers.length;
   var count = 0;
   for (var l=0; l<(layers-1); l++) {
      var L = network.layers[l];
      console.log("Layer "+l+": "+L.layer_type);
      var neurons = network.layers[l].out_act.w.length;
      var next = network.layers[l+1].out_act.w.length;

      var arr = [];
      for (var n=0; n<neurons; n++) {
         arr.push(points[count]);
         count++;
      }
      data.network.push({"layer":l, "neuron":neurons, "type":L.layer_type, "points":arr});
   }

   var str = JSON.stringify(data, null, 2); 
    
   var fs = require('fs');
   fs.writeFile(filename, str, function(err) {
      if (err) {
         console.log(err);
      } else {
         console.log("json saved to "+filename);
      }
   });
}

////////// write out json file of data for sankey

data_to_sankey(net, 'data/tsne_sankey.json');

function data_to_sankey(network, filename) {

   var data = {"nodes":[], "links":[], "meta":[]};

   // for each layer, add nodes for each neuron
   var count = 0;
   var layers = network.layers.length;
   //console.log("total layers; "+layers);
   for (var l=0; l<(layers-1); l++) {
      var L = network.layers[l];
      console.log("Layer "+l+": "+L.layer_type);
      var neurons = network.layers[l].out_act.w.length;
      var next = network.layers[l+1].out_act.w.length;
      //console.log("layer: "+l);
      
      for (var n=0; n<neurons; n++) {
         var name = "L"+l+"N"+n;
         //data.nodes.push({"names":name, "l":l, "n":n, "points":points[count+n]});
         data.nodes.push({"name":name,"value":points[count+n]});

         data.meta.push({"size":(n+1)*100, "pos":points[count+n]});
         //console.log("\tneuron: "+(count+n));
         if (l<(layers-2)) {
            for (var n2=0; n2<next; n2++) {
               //console.log("\t\tto neuron: "+(count+neurons+n2));
               //console.log(count+neurons+n2);
               data.links.push({"source":count+n,
               "target":count+neurons+n2,
               "value":1-Math.abs(points[count+neurons+n2]-points[count+n]),
               "v1":points[count+n],
               "v2":points[count+neurons+n2]});
            }
         }
      }
      count = count + neurons;
   }

/*
   for (var l=0; l<(layers-1); l++) {
      for (var n=0; n<neurons; n++) {
         data.nodes.push({"size":(n+1)*100});
      }
   }
   */

   // get the link data too
   
   var str = JSON.stringify(data, null, 2); 
    
   var fs = require('fs');
   fs.writeFile(filename, str, function(err) {
      if (err) {
         console.log(err);
      } else {
         console.log("json saved to "+filename);
      }
   });
}
