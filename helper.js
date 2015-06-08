
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
    var mapRange = function(from, to, s) {
               return to[0] + (s - from[0]) * (to[1] - to[0]) / (from[1] - from[0]);
            };
    helper.prototype = {

       normalizeBetweenLayers: function(network, points, collapse_input) {
          var count = 0;
          var idx=0;

          var layers = network.layers.length;
          var links = [];
          var start = 0;
          var end = 0;
          for (var l=0; l<(layers); l++) {
             var L = network.layers[l];
             var neurons = L.out_depth*L.out_sx*L.out_sy;
             if (L.layer_type != "fc" && L.layer_type != "conv") {
                // Good current layer
                console.log("Layer "+l+": "+L.layer_type+" this: "+neurons);
                var name = "L"+l+"N"+n;
                var next_l = 1;
                if (l < layers-1) {
                   while (network.layers[l+next_l].layer_type == "fc"
                   || network.layers[l+next_l].layer_type == "conv") {
                      next_l++;
                   }
                   var nl = l+next_l;
                   var Ln = network.layers[nl];
                   var next = Ln.out_depth*Ln.out_sx*Ln.out_sy;
                   if (L.layer_type == "input" && collapse_input > 0) {
                      neurons = collapse_input;
                   }
                   start = links.length;
                   end = start;
                   for (var n=0; n<neurons; n++) {
                      if (n==0) {
                         console.log("\tto Layer: "+nl+" type: "+Ln.layer_type+" next: "+next);
                      }
                      for (var n2=0; n2<next; n2++) {
                         //var val = 1.0-Math.sqrt((points[count+neurons+n2]-points[count+n])^2);
                         var val = 1.0-Math.abs(points[count+neurons+n2]-points[count+n]);
                         //console.log("a: "+points[count+neurons+n2]+" b: "+points[count+n]+" val: "+(points[count+neurons+n2]-points[count+n])^2);
                         links.push(val)
                         end++;
                      }
                   }
                }
                var min = Math.min.apply(null, links.slice(start, end));
                var max = Math.max.apply(null, links.slice(start, end));

                console.log("BETWEEN layers "+l+" "+neurons+" from: "+start+" :: "+end);
                console.log("\told min: "+min+" max: "+max);
                for (var n=start; n<end; n++) {
                   links[n] = mapRange([min, max], [0, 1], links[n]);
                }
                min = Math.min.apply(null, links.slice(start, end));
                max = Math.max.apply(null, links.slice(start, end));
                console.log("\tnew min: "+min+" max: "+max);

                console.log("\t"+count+" neurons");
                count = count + neurons;
             }
          }
          console.log("Links data: "+links.length);

          return links;
       },

       normalizeByLayers: function(network, data, start, end) {
          var layers = network.layers.length;
          var x = 0;
          for (var l=start; l<end; l++) {
             var L = network.layers[l];
             if (L.layer_type != "fc" && L.layer_type != "conv" && L.layer_type != "regression") {
                var neurons = L.out_depth*L.out_sx*L.out_sy;

                var l_norm = 0;
                var min = Math.min.apply(null, data.slice(x,x+neurons));
                var max = Math.max.apply(null, data.slice(x,x+neurons));

                //console.log(data.slice(x,x+neurons)+" min: "+min);
                console.log("By Layers:"+ l+" "+neurons+" min: "+min+" max: "+max);
                for (var n=0; n<neurons; n++) {
                   data[x+n] = mapRange([min, max], [0, 1], data[x+n]);
                }

                min = Math.min.apply(null, data.slice(x,x+neurons));
                max = Math.max.apply(null, data.slice(x,x+neurons));
                console.log("\tmin: "+min+" max: "+max);

                x = x+neurons;
             }
          }
          return data;
       },

       dataToSankey: function(network, points, links, opts, filename, collapse_input, slim_thresh) {

          var data = {"nodes":[], "links":[], "meta":[], "opt":opts};

          // for each layer, add nodes for each neuron
          var count = 0;
          var layer_count = 0;
          var idx=0;
          var used=0;
          var layers = network.layers.length;
          for (var l=0; l<(layers); l++) {
             var L = network.layers[l];
             var neurons = L.out_depth*L.out_sx*L.out_sy;
             if (L.layer_type != "fc" && L.layer_type != "conv") {
                console.log("Layer "+l+": "+L.layer_type+" this: "+neurons);
                // Good current layer
                if (L.layer_type == "input" && collapse_input > 0) {
                   neurons = collapse_input;
                }
                //else {
                   for (var n=0; n<neurons; n++) {
                      var name = "L"+l+"N"+n;
                      //console.log(name + " "+points[count+n]+" "+count+" "+n+ " ");
                      data.nodes.push({"name":name,"layer":L.layer_type,"col":layer_count,"value":points[count+n],"num":count+n});
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
                            if (L.layer_type == "input" && collapse_input > 0) {
                               var val = 0.6;
                               data.links.push({"source":count+n,
                                  "target":count+neurons+n2,
                                  "value":val,
                                  "v1":points[count+n],
                                  "v2":points[count+neurons+n2],
                               "idx":idx});
                               used++;
                               idx++;
                            }
                            else if (Ln.layer_type == "regression") {
                               //var val = 1.0-Math.abs(points[count+neurons+n2]-points[count+n]);
                               var val = links[idx]; //1.0-Math.sqrt((points[count+neurons+n2]-points[count+n])^2);
                               if (val > slim_thresh) {
                                  data.links.push({"source":count+n,
                                     "target":count+neurons+n2,
                                     "value":val,
                                     "v1":points[count+n],
                                     "v2":points[count+neurons+n2],
                                  "idx":idx});
                                  used++;
                               }
                               idx++;
                            }
                            else {
                               //var val = 1.0-Math.abs(points[count+neurons+n2]-points[count+n]);
                               //var val = Math.abs(points[count+neurons+n2]-points[count+n]);
                               var val = 0.5; //links[idx]; //1.0-Math.sqrt((points[count+neurons+n2]-points[count+n])^2);
                               if (val > slim_thresh) {
                                  data.links.push({"source":count+n,
                                     "target":count+neurons+n2,
                                     "value":val,
                                     "v1":points[count+n],
                                     "v2":points[count+neurons+n2],
                                  "idx":idx});
                                  used++;
                               }
                               idx++;
                            }
                         }
                      }
                   }
                   //}
                   console.log("\t"+count+" neurons");
                   count = count + neurons;
                   layer_count++;
             }
          }
          count += next;
          console.log("\t"+count+" neurons TOTAL");
          console.log("\t"+idx+" connections TOTAL");
          console.log("\t"+links.length+" normalized links");
          console.log("\t"+used+" connections used");

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
