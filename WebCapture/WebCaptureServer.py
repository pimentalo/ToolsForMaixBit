import http.server

PORT = 8888
server_address = ("", PORT)

server = http.server.HTTPServer
handler = http.server.CGIHTTPRequestHandler
handler.cgi_directories = ["/"]
print("Server started, you can access it on http://%s:%d/index.py" % ("localhost", PORT))

httpd = server(server_address, handler)
httpd.serve_forever()
