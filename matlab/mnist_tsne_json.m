
x=100;
y=neurons;
num_input = 1;
num_output=1;
perplexity=10;
filename='tsne_tmp.json';

% no better way of getting data in yet other than ctrl-v ... :(
data=[]; 
d = reshape(data, [x,y]);
d_mid=d(:,num_input+1:neurons-num_output);

m_mid=tsne(d_mid', [], 2, [], perplexity); % get 2-d tsne to plot
plot(m_mid(:,1), m_mid(:,2),'.')

%% after testing tsne result, send to json

mid_n=tsne(d_mid', [], 1, [], perplexity);

input=rand(num_inputs,1);
output=rand(num_output,1);

out = [input;mid_n;output];

j=savejson('points',out,filename);

% normalize in javascript
