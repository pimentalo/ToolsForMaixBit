import cgi 
import cgitb
import io
import sys

cgitb.enable() # Enable error report

form = cgi.FieldStorage()
print("Content-type: text/html; charset=utf-8\n")
#print(form.getvalue("name"))

file = None
try :
    with io.open("html/prepareDataset.html",'r',encoding='utf8') as f:
        html = f.read()
except:
    print("Unexpected error:", sys.exc_info()[0])
    html = sys.exc_info()[0]
    raise

print(html)
