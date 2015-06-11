# Neural Network Behavior 

Kendall Lowrey -- klowrey

Tam Nguyen -- nttam15

Artificial Neural Networks have traditionally been treated as black boxes, both in their development and in their use. We present a method to discover the internal structure of neural networks by visualizing activation properties of the network with respect to input data: co-activations of multiple neurons. Our method combines statistical analysis techniques with a modified Sankey Diagram to show flow of data through neural networks unlike previous visualizations methods. Implications for this technique beyond behavioral and structural visualization include the optimization of an artificial neural network through parameter reduction and further understanding of their processing.


# Training Networks for Data Collection

The only network we can train simply is the XOR network. We run the tsne_xor.js code with nodejs, which usings some functions from helper.js, scripts/tsne.js, and scripts/convnet-min.js, the last two of which are provided by https://github.com/karpathy

The resulting data files ( written to data/xor--.json ) are then read by the index.html front-end and rendered to svg through d3js.

# Paper & Poster

Are located under the final directory


# Work breakdown and Research Process

Research for this project began with wanting to understand the inner workings of neural networks from a non-math based perspective. This required previous experience with them, as well as other optimization techniques. tSNE cam up as the best way of visualizing the co-activations of neural networks and was research as the best method. 

Additionally, we strove to understand what other methods were used previously in visualizing neural networks and they are presented in our paper contribution. We found that this kind of visualization has, to the best of our knowledge, not been attempted before.

Work consisted of a significant amount of support software beyond just visualization code as we needed to support multiple different neural networks all operating on different data inputs. This required Javascript, Python, and Matlab. 

