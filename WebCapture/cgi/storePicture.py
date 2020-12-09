import cgi 
import cgitb
import os
import sys
import base64
import http

cgitb.enable() # Enable error report

form = cgi.FieldStorage()
print("Content-type: text/html; charset=utf-8\n")

sDataset = form.getvalue("dataset")
sPicture = form.getvalue("picture")
sImageFile = "./datasets/" + sDataset + "/" + "0000000.png"

try :
    image = base64.b64decode(sPicture)

    with io.open(sImageFile,'w') as f:
        html = f.write(image)
    print(http.HTTPStatus.OK)
except:
    print("Unexpected error:", sys.exc_info()[0])
    html = sys.exc_info()[0]
    raise

print(http.HTTPStatus.INTERNAL_SERVER_ERROR)
