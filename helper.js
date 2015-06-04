
var helperjs = helperjs || { REVISION: 'ALPHA' };

(function(global) {
   "use strict";

   // syntax sugar from karpathy
   var getopt = function(opt, field, defaultval) {
      if(opt.hasOwnProperty(field)) {
         return opt[field];
      } else {
         return defaultval;
      }
   }

   var helper = function(opt) {
      var opt = opt || {};
      this.perplexity = getopt(opt, "perplexity", 30); // effective number of nearest neighbors
      this.dim = getopt(opt, "dim", 2); // by default 2-D tSNE
      this.epsilon = getopt(opt, "epsilon", 10); // learning rate

      this.iter = 0;
   }

   helper.prototype = {

      normalizeByLayers: function(network, data) {
         var layers = network.layers.length;
         var x = 0;
         for (var l=0; l<(layers); l++) {
            var L = network.layers[l];
            if (L.layer_type != "fc" && L.layer_type != "conv") {
               var neurons = L.out_depth*L.out_sx*L.out_sy;

               var l_norm = 0;
               var min = Math.min.apply(null, data.slice(x,x+neurons));
               var div = Math.max.apply(null, data.slice(x,x+neurons))-min;

               //console.log(data.slice(x,x+neurons)+" min: "+min);
               console.log(l+" layer min: "+min+" max: "+(div+min));
               for (var n=0; n<neurons; n++) {
                  if (Math.abs(div) > 1e-9) {
                     data[x+n] = 0.9* ((data[x+n]-min) / (div)) + 0.05;
                  }
                  //l_norm += points[n] * points[n];
               }

               min = Math.min.apply(null, data.slice(x,x+neurons));
               div = Math.max.apply(null, data.slice(x,x+neurons))-min;
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
         return data;
      },

      dataToSankey: function(network, points, opts, filename, collapse_input) {

         var data = {"nodes":[], "links":[], "meta":[], "opt":opts};

         // for each layer, add nodes for each neuron
         var count = 0;
         var idx=0;
         var layers = network.layers.length;
         //console.log("total layers; "+layers);
         for (var l=0; l<(layers); l++) {
            var L = network.layers[l];
            var neurons = L.out_depth*L.out_sx*L.out_sy;
            if (L.layer_type != "fc" && L.layer_type != "conv") {
               console.log("Layer "+l+": "+L.layer_type+" this: "+neurons);
               // Good current layer
               if (L.layer_type == "input" && collapse_input == true) {
                  neurons = 1;
               }
               else {
                  for (var n=0; n<neurons; n++) {
                     var name = "L"+l+"N"+n;
                     data.nodes.push({"name":name,"layer":L.layer_type,"value":points[count+n], "num":count+n});
                     //data.meta.push({"size":Math.abs(network.layers[l].out_act.w[n]),"pos":points[count+n]});
                     data.meta.push({"size":10+n%5,"pos":points[count+n]});
                     var next_l = 1;
                     if (l < layers-1) {
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
                        for (var n2=0; n2<next; n2++) {
                           var val = 1.1-Math.abs(points[count+neurons+n2]-points[count+n]);
                           data.links.push({"source":count+n,
                              "target":count+neurons+n2,
                              //"value":Math.sqrt(points[count+neurons+n2]^2+points[count+n]^2),
                              "value":val,
                              "v1":points[count+n],
                              "v2":points[count+neurons+n2],
                              "idx":idx});
                           idx++;
                        }
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
   }
   global.helper = helper;
})(helperjs);

// export the library to window, or to module in nodejs
(function(lib) {
   "use strict";
   if (typeof module === "undefined" || typeof module.exports === "undefined") {
      window.helperjs = lib; // in ordinary browser attach library to window
   } else {
      module.exports = lib; // in nodejs
   }
})(helperjs);
