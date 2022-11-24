import json

fileName = '/home/simong/AcquPy/aux/taggerNew.json'

f = open('/home/simong/AcquPy/aux/tagger.dat', 'r')
referenceADC = [927,1055,1183]
channel = 0

with open(fileName,'w') as outfile:

    data = {}
    data['detector']   = 'tagger'
    data['calEq']      = "return [(y['raw']-y['offset']-ref[int(y['referenceID'])])*y['scale'] for y in x]"
    data['channels']   = []
    data['references'] = []
    
    for line in f:
        if('#' in line): continue
        if('Element:' not in line): continue
        columns = line.split()

        datum = {}
        datum['channel']      = channel
        datum['adc']          = int(columns[6].split('M')[0])
        datum['energy']       = float(columns[14])
        datum['ewidth']       = float(columns[15])
        datum['scale']        = float(columns[10])
        datum['offset']       = float(columns[9])
        datum['referenceID']  = min([i for i,x in enumerate(referenceADC) if x >=datum['adc']])
        datum['raw']          = []
        datum['value']        = []

        data['channels'].append(datum)
        
        channel    += 1

    for ref in referenceADC:
        datum = {}
        datum['adc'] = int(ref)
        datum['raw'] = []
        data['references'].append(datum)
    

    print(data)
    json.dump(data, outfile, indent=2)
