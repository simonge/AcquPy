import json

#inFile   = '/home/simong/git/acqu/acqu_user/data_MC12/FP_1508.RecPol.09.08.dat'
#tag = ''
#references = [[1400]]
#beamE      = 1508.0

inFile     = '/home/simong/AcquPy/aux/tagger.dat'
tag        = 'New'
references = [[927,1055,1183]]
beamE      = 1557.0

fileName = '/home/simong/AcquPy/aux/tagger'+tag+'.json'

parameters = ['Time']
#equations  = ["[(y['raw']-y['offset']-ref[0])*y['scale'] for y in x]"]
equations  = ["[(y['raw']-y['offset']-ref[int(y['referenceID'])])*y['scale'] for y in x]"]
ignore     = []

channel = 0

with open(fileName,'w') as outfile:
    f = open(inFile,'r')

    channel = 0
    data = {}
    data['detector']   = 'Tagger'
    
    data['parameters'] = parameters
    data['calEq']      = equations
    data['references'] = references
    data['ignore']     = ignore
    data['channels']   = []
    
    
    for line in f:
        if('#' in line): continue
        if('Element:' not in line): continue
        columns = line.split()

        datum = {}
        datum['channel']      = channel
        datum['adc']          = [int(columns[6].split('M')[0])]
        datum['electronE']    = float(columns[14])
        datum['Energy']       = beamE-datum['electronE']
        datum['ewidth']       = float(columns[15])
        datum['scale']        = float(columns[10])
        datum['offset']       = float(columns[9])
        datum['referenceID']  = min([i for i,x in enumerate(references[0]) if x>=datum['adc'][0]])
        #datum['referenceID']  = int(references[0][0])
        datum['neighbours']   = [channel-1,channel+1]
        datum['position']     = [channel,datum['electronE'],datum['Energy']]
        datum['raw']          = []

        data['channels'].append(datum)
        
        channel    += 1

    data['channels'][0]['neighbours'] = [data['channels'][0]['neighbours'][1]]
    data['channels'][channel-1]['neighbours'] = [data['channels'][channel-1]['neighbours'][0]]

    #print(data)
    json.dump(data, outfile, indent=2)
