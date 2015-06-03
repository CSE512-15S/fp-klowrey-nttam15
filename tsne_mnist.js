/* global labels */
var tsnejs = require('./scripts/tsne');
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
var from_file = require('./matlab/simple_mnist_v2.json'); //tsne data from matlab
var points = from_file.points
console.log("Points: "+points.length);

// NORMALIZE the tsne outputs
//console.log(points);
console.log("Normalizing by layers");
var layers = net.layers.length;
var x = 0;
for (var l=0; l<(layers); l++) {
   var L = net.layers[l];
   if (L.layer_type != "fc" && L.layer_type != "conv") {
      var neurons = L.out_depth*L.out_sx*L.out_sy;

      var l_norm = 0;
      var min = Math.min.apply(null, points.slice(x,x+neurons));
      var div = Math.max.apply(null, points.slice(x,x+neurons))-min;

      //console.log(points.slice(x,x+neurons)+" min: "+min);
      console.log(l+" layer min: "+min+" max: "+(div+min));
      for (var n=0; n<neurons; n++) {
         if (Math.abs(div) > 1e-9) {
            points[x+n] = (points[x+n]-min) / (div);
         }
            //l_norm += points[n] * points[n];
      }
      
      min = Math.min.apply(null, points.slice(x,x+neurons));
      div = Math.max.apply(null, points.slice(x,x+neurons))-min;
      console.log("\tmin: "+min+" max: "+(div+min));
      /*
      var mag = Math.sqrt(l_norm); 
      for (var n=0; n<neurons; n++) {
         var val = points[n] / mag;
         val = val + min;
         points[n] = val / 2.0;
      }
      */
      x = x+neurons;
   }
}

//console.log(points);


for (var l=0; l<(layers-1); l++) {
   var L = net.layers[l];
   console.log("Layer "+l+": "+L.layer_type);
}
////////// write out json file of data for sankey

data_to_sankey(net, 'data/mnist_sankey_v2.json');

function data_to_sankey(network, filename) {

   var data = {"nodes":[], "links":[], "meta":[]};

   // for each layer, add nodes for each neuron
   var count = 0;
   var layers = network.layers.length;
   //console.log("total layers; "+layers);
   for (var l=0; l<(layers-1); l++) {
      var L = network.layers[l];
      var neurons = L.out_depth*L.out_sx*L.out_sy;
      if (L.layer_type != "fc" && L.layer_type != "conv") {
      console.log("Layer "+l+": "+L.layer_type+" this: "+neurons);
         // Good current layer
         for (var n=0; n<neurons; n++) {
            var name = "L"+l+"N"+n;
            data.nodes.push({"name":name,"layer":L.layer_type,"value":points[count+n]});
            //data.meta.push({"size":Math.abs(network.layers[l].out_act.w[n]),"pos":points[count+n]});
            data.meta.push({"size":10+n%5,"pos":points[count+n]});
            var next_l = 1;
            while (network.layers[l+next_l].layer_type == "fc"
            || network.layers[l+next_l].layer_type == "conv") {
               next_l++;
            }
            var nl = l+next_l;
            var Ln = network.layers[nl];
            var next = Ln.out_depth*Ln.out_sx*Ln.out_sy;
            if (n==0) {
               console.log("\tto Layer: "+nl+" type: "+Ln.layer_type+" next: "+next);
            }
            if (nl < layers-1) {
               for (var n2=0; n2<next; n2++) {
                  data.links.push({"source":count+n,
                     "target":count+neurons+n2,
                     //"value":Math.sqrt(points[count+neurons+n2]^2+points[count+n]^2),
                     "value":1-Math.abs(points[count+neurons+n2]-points[count+n]),
                     "v1":points[count+n],
                  "v2":points[count+neurons+n2]});
               }
            }
         }
      console.log("\t"+count+" neurons");
      count = count + neurons;
      }
   }
   count += next;
   console.log("\t"+count+" neurons TOTAL");
   
   var fs = require('fs');
   var str = JSON.stringify(data, null, 2); 
   fs.writeFile(filename, str, function(err) {
      if (err) {
         console.log(err);
      } else {
         console.log("json saved to "+filename);
      }
   });
}

