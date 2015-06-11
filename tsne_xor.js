/* global labels */
var tsnejs = require('./scripts/tsne');
var convnetjs = require('./scripts/convnet-min');


//var output_type = 'softmax'
var output_type = 'regression'

// make layers
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:3});
layer_defs.push({type:'fc', num_neurons:12, activation:'relu'});
layer_defs.push({type:'fc', num_neurons:12, activation:'relu'});

if (output_type == 'softmax') {
   layer_defs.push({type:'softmax', num_classes:2});
}
else {
   layer_defs.push({type:'regression', num_neurons:1});
}

var small_l = []
small_l.push({type:'input', out_sx:1, out_sy:1, out_depth:3});
small_l.push({type:'fc', num_neurons:6, activation:'relu'});
small_l.push({type:'fc', num_neurons:5, activation:'relu'});
if (output_type == 'softmax') {
   small_l.push({type:'softmax', num_classes:2});
}
else {
   small_l.push({type:'regression', num_neurons:1});
}

// make network from layers above
var net = new convnetjs.Net();
var no_train = new convnetjs.Net();
var s_net = new convnetjs.Net();

net.makeLayers(layer_defs);
no_train.makeLayers(layer_defs);
s_net.makeLayers(small_l);

var trainer = new convnetjs.Trainer(net, 
              {method: 'adadelta', batch_size:10, l2_decay:0.001});
var notrainer = new convnetjs.Trainer(no_train, 
              {method: 'adadelta', batch_size:10, l2_decay:0.001});
var strainer = new convnetjs.Trainer(s_net, 
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
if (output_type == 'softmax') {
data.push([0.0, 0.0, 0.0]); labels.push(0);
data.push([0.0, 1.0, 0.0]); labels.push(1);
data.push([1.0, 0.0, 0.0]); labels.push(1);
data.push([1.0, 1.0, 0.0]); labels.push(0);
data.push([0.0, 0.0, 1.0]); labels.push(1);
data.push([0.0, 1.0, 1.0]); labels.push(0);
data.push([1.0, 0.0, 1.0]); labels.push(0);
data.push([1.0, 1.0, 1.0]); labels.push(1);
}
else{
data.push([0.0, 0.0, 0.0]); labels.push([0]);
data.push([0.0, 1.0, 0.0]); labels.push([1]);
data.push([1.0, 0.0, 0.0]); labels.push([1]);
data.push([1.0, 1.0, 0.0]); labels.push([0]);
data.push([0.0, 0.0, 1.0]); labels.push([1]);
data.push([0.0, 1.0, 1.0]); labels.push([0]);
data.push([1.0, 0.0, 1.0]); labels.push([0]);
data.push([1.0, 1.0, 1.0]); labels.push([1]);
}

var N = data.length;

var x = new convnetjs.Vol(1,1,2);

function train_network(train, d, l, iter) {
   var start = new Date().getTime();
   var avloss = 0.0;
   for(var iters=0;iters<iter;iters++) {
      for(var ix=0;ix<N;ix++) {
         x.w = d[ix];
         var stats = train.train(x, l[ix]);
         avloss += stats.loss;
      }
   }
   avloss /= N*iters;

   var end = new Date().getTime();
   var time = end - start;

   console.log('loss = ' + avloss + ', 100 cycles through data in ' + time + 'ms');
}


function bad_train_network(train, d, l, iter) {
   var start = new Date().getTime();
   var avloss = 0.0;
   for(var iters=0;iters<iter;iters++) {
      for(var ix=0;ix<N;ix++) {
         x.w = d[ix];
         var stats = train.train(x, Math.round(Math.random()));
         avloss += stats.loss;
      }
   }
   avloss /= N*iters;

   var end = new Date().getTime();
   var time = end - start;

   console.log('loss = ' + avloss + ', 100 cycles through data in ' + time + 'ms');
}

var iter = 4000;
train_network(trainer, data, labels, iter);
bad_train_network(notrainer, data, iter);
train_network(strainer,data, labels, iter);

console.log("\nBig Trained");
for(var ix=0;ix<N;ix++) {
   x.w = data[ix];
   var predicted_values = net.forward(x);
   console.log('in: ' + data[ix]+' goal: '+labels[ix]+' out: '+predicted_values.w[0]+' '+predicted_values.w[1]);
}
console.log("\nBig Untrained");
for(var ix=0;ix<N;ix++) {
   x.w = data[ix];
   var predicted_values = no_train.forward(x);
   console.log('in: ' + data[ix]+' goal: '+labels[ix]+' out: '+predicted_values.w[0]+' '+predicted_values.w[1]);
}
console.log("\nSmall Trained");
for(var ix=0;ix<N;ix++) {
   x.w = data[ix];
   var predicted_values = s_net.forward(x);
   console.log('in: ' + data[ix]+' goal: '+labels[ix]+' out: '+predicted_values.w[0]+' '+predicted_values.w[1]);
}

var test_data = [];
for (var i=0; i<4; i++) {
   var idx = Math.floor(Math.random() * data.length);
   test_data.push(data[idx]);
}

var tsne_data = get_coactivation_data(net, test_data);
var raw_data = get_coactivation_data(no_train, test_data);
var small_tsne= get_coactivation_data(s_net, test_data);

function layer_data(network, act) {
   var layers = network.layers.length;
   var count = 0;
   for (var l=1; l<(layers-1); l++) {
      var L = network.layers[l];
      if (L.layer_type != "fc" && L.layer_type != "conv") {
         var neurons = network.layers[l].out_act.w.length;
         for (var nw=0; nw<neurons; nw++) {
            weight = network.layers[l].out_act.w[nw];
            act[count].push(weight);
            count = count+1;
         }
      }
   }
}

// get coactivations with test dataset test_d
function get_coactivation_data(network, test_d) {
   var N = test_d.length;
   var layers = network.layers.length;
   var activations = [];
   var count = 0;
   // skip first and last layer
   for (var l=1; l<(layers-1); l++) {
      var L = network.layers[l];
      if (L.layer_type != "fc" && L.layer_type != "conv") {
         var neurons = network.layers[l].out_act.w.length;
         for (var nw=0; nw<(neurons); nw++) {
            activations[count] = [];
            count = count+1;
         }
      }
   }
   console.log(count+" total neurons");

   // for each test input, net forward...
   for (var ix=0;ix<N;ix++) {
         x.w = test_d[ix];
         var predicted_values = network.forward(x);
         // then get each layer of data
         layer_data(network, activations);
   }
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

var helperjs = require('./helper.js');
var helper = new helperjs.helper();

var opt = {epsilon: 10, perplexity: 60, dim: 1};

function get_tsne(options, data, iters) {
   var tsne = new tsnejs.tSNE(options); // create a tSNE instance
   //tsne.initDataDist(data);
   tsne.initDataRaw(data);
   if (iters <= 0) {
      return null;
   }
   for(var k = 0; k < iters; k++) {
      tsne.step(); // every time you call this, solution gets better
   }
   var d = tsne.getSolution();
   var arr = [];
   for (var i=0; i<d.length; i++) {
      arr.push(d[i][0]);
   }
   return arr;
}

console.log("TSNE input:");
console.log(tsne_data.length+" by "+tsne_data[0].length +" test inputs");
console.log("TSNE Output:");
var points = get_tsne(opt, tsne_data, 1000);
var notrain= get_tsne(opt, raw_data, 500);
var smaller= get_tsne(opt, small_tsne, 500);

function linspace(a,b,n) {
   if(typeof n === "undefined") n = Math.max(Math.round(b-a)+1,1);
   if(n<2) { return n===1?[a]:[]; }
   var i,ret = Array(n);
   n--;
   for(i=n;i>=0;i--) { ret[i] = (i*b+(n-i)*a)/n; }
   return ret;
}

points = sandwich_in_out(net, points);
notrain= sandwich_in_out(no_train,notrain);
smaller= sandwich_in_out(s_net,smaller);

// sandwich linspace for inputs and outputs
function sandwich_in_out(network, data) {
   var iL = network.layers[0];
   var oL = network.layers[network.layers.length-1];

   var n_i = iL.out_depth*iL.out_sx*iL.out_sy;
   var n_o = oL.out_depth*oL.out_sx*oL.out_sy;

   data = linspace(0,1,n_i).concat(data);
   if (n_o == 1) {
      data= data.concat([0.5]);
   } else {
      data= data.concat(linspace(0,1,n_o));
   }

   console.log(data);
   return data;
   // NORMALIZE the tsne outputs
}


l_points = helper.normalizeBetweenLayers(net, points, -1);
l_notrain= helper.normalizeBetweenLayers(no_train,notrain, -1);
l_smaller= helper.normalizeBetweenLayers(s_net,smaller, -1);

points = helper.normalizeByLayers(net, points, 0, 7);
notrain= helper.normalizeByLayers(no_train,notrain, 0, 7);
smaller= helper.normalizeByLayers(s_net,smaller, 0, 7);


console.log("TSNE normalized");
console.log(points);
console.log("TSNE normalized");
console.log(notrain);
console.log("TSNE normalized");
console.log(smaller);

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

var net_opt ={"name":'xor_net', 'node_scale':10, 'link_scale':20, 'link_opcty':2,
    'nodeWidth':24,
    'nodePadding':0};
var collapse_input = false;
var slim_thresh = 0.0;
console.log("links: "+l_points.length);
helper.dataToSankey(net, points, l_points, net_opt,'data/xor_sankey.json', collapse_input, slim_thresh);
helper.dataToSankey(no_train,notrain, l_notrain, net_opt,'data/xor_notrain_sankey.json', collapse_input, slim_thresh);
helper.dataToSankey(s_net,smaller, l_smaller, net_opt,'data/xor_smaller_sankey.json', collapse_input, slim_thresh);


