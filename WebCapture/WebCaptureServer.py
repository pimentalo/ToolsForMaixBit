import http.server

PORT = 8888
server_address = ("", PORT)

handler = http.server.CGIHTTPRequestHandler
handler.cgi_directories = ["/cgi"]

with http.server.HTTPServer(("", PORT), handler) as httpd:
    print("Server started, you can access it on http://%s:%d/index.py" % ("localhost", PORT))
    httpd.serve_forever()
