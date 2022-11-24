import pandas as pd
from pandas.io.json import json_normalize
import json

fileName = '/home/simong/AcquPy/aux/tagger.json'
#fileName = '/home/louise/coatjava_20181112/etc/bankdefs/hipo/CLAS6EVENT.json'
infile   = open(fileName)

data = json.load(infile)

print(data)
#data = [{'id': 1, 'name': {'first': 'Coleen', 'last': 'Volk'}},
#        {'name': {'given': 'Mose', 'family': 'Regner'}},
#        {'id': 2, 'name': 'Faye Raker'}]

#data = pd.read_json(fileName)

df = json_normalize(data,'channels',['detector','calEq']).set_index('adc')
print(df)

