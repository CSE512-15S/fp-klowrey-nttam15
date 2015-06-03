
var connect = require('connect');
var serveStatic = require('serve-static');
connect().use(serveStatic(__dirname)).listen(8080);




/*
var http = require('http'),
      fs = require('fs'),
     url = require('url');

http.createServer(function(request, response){
    var path = url.parse(request.url).pathname;
    if(path=="/getdata"){
        console.log("request recieved");
        var string = choices[Math.floor(Math.random()*choices.length)];
        console.log("string '" + string + "' chosen");
        response.writeHead(200, {"Content-Type": "text/plain"});
        response.end(string);
        console.log("string sent");
        to_file(request.data);
    }else{
        fs.readFile('./index.html', function(err, file) {  
            if(err) {  
                // write an error response or nothing here  
                return;  
            }  
            response.writeHead(200, { 'Content-Type': 'text/html' });  
            response.end(file, "utf-8");  
        });
    }
}).listen(8080);
console.log("server initialized");

function to_file(json, filename) {
   fs.writeFile(filename, JSON.stringify(json), function(err) {
      if (err) {
         console.log(err);
      } else {
         console.log("json saved to "+filename);
      }
   });
}
*/
// wait for 
