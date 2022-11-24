import json

fileName = '/home/simong/AcquPy/aux/paddle.json'

references = [[]]
parameters = ['Energy']
equations  = ["x['raw']"]
ignore     = []
adcs       = [1055]        

with open(fileName,'w') as outfile:

    data = {}
    data['detector']   = 'Paddle'
    data['parameters'] = parameters
    data['calEq']      = equations
    data['references'] = references
    data['ignore']     = ignore
    data['channels']   = []
       
    channel = 0
    
    for adc in adcs:

        datum = {}
        datum['channel']      = channel
        datum['adc']          = [int(adc)]
        datum['scale']        = 1
        datum['position']     = [-50,-50,-50]
        datum['neighbours']   = []
        datum['raw']          = []

        data['channels'].append(datum)
            
    print(data)
    json.dump(data, outfile, indent=2)
