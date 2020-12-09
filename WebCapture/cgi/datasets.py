#
# Allow to create, list, delete datasets
# 
# Each dataset is just a directory in the "datasets" directory
#
import cgi 
import cgitb
import sys
import os
import json
import http.server


cgitb.enable() # Enable error report

form = cgi.FieldStorage()
print("Content-type: text/html; charset=utf-8\n")
datasetsPath = "./datasets"

sVerb = ""
if "verb" in form:
    sVerb = form.getvalue("verb")


# Get: list datasets
try :
    if (not os.path.exists(datasetsPath)):
        os.mkdir(datasetsPath)
    if sVerb == "" or sVerb == "list":
        dirs = os.listdir(datasetsPath)
        print(json.dumps(dirs))
    elif sVerb == "create":
        sName = form.getvalue("name")
        sDatasetPath = datasetsPath + "/" + sName
        if (os.path.exists(datasetsPath)):
            print('Status: 403 Forbidden\r\n\r\n')
        else:
            os.mkdir(sDatasetPath)

except:
    print("Unexpected error:", sys.exc_info()[0])
    raise