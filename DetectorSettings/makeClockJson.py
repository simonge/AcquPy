import json

fileName = '/home/simong/AcquPy/aux/clock.json'

references = [[301]]
parameters = ['Time']
equations  = ["x['raw']+65536*ref[0]"]
ignore     = []
adcs       = [300]        

with open(fileName,'w') as outfile:

    data = {}
    data['detector']   = 'Clock'
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
        datum['scale']        = float(2.5)
        datum['position']     = [50,50,50]
        datum['neighbours']   = []
        datum['raw']          = []

        data['channels'].append(datum)
            
    print(data)
    json.dump(data, outfile, indent=2)
