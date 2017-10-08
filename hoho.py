import json
import pprint

data = {
   'name' : 'ACME',
   'shares' : 100,
   'price' : 542.23
}

json_str = json.dumps(data)
# pp = pprint.PrettyPrinter(indent=4)
print(json_str)