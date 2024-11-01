from requests.auth import HTTPBasicAuth
import requests
import json


url		= "http://192.168.56.107:8500/refusal_check/2bclpy6R6wDboaVlmcGWkSm0GrBGCoBG/Пл2277680"





r = requests.get(url, '', )



#print(r.status_code, r.reason)
print(r.text)