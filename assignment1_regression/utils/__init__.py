import json
def readjson(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data