import json

#inFile = '/home/simong/git/acqu/acqu_user/data_MC12/BaF2.12.01.07a'
#tag = 'Original'

#inFile = '/home/simong/acqu/AcquData/data_MC12/BaF2_boabie.dat'
#tag = 'boabie'

#inFile = 'home/simong/git/acqu/acqu_user/data.general/Detector-BaF2-PbWO4-S-2008.04.dat'
#tag    = '2008'

#inFile = '/home/simong/acqu-ROOT6/acqu_user/data.2016.06/Detector-BaF2.dat'
#tag = 'New'

inFile = '/home/simong/AcquPy/aux/TAPSNewest.dat'
tag = 'Newest'

#inFile = '/home/simong/AcquPy/aux/BaF2_PWO.dat'
#tag = 'Dom'

fileName = '/home/simong/AcquPy/aux/TAPS'+tag+'.json'

parameters = ['Energy','EnergySG','Time']
equations  = ["[float(y['raw'])*y['scale'][i]+y['offset'][i] for y in x]","[float(y['raw'])*y['scale'][i]+y['offset'][i] for y in x]","[y['raw']*y['scale'][i]-y['offset'][i] for y in x]"]
references = [[],[],[]]
ignore     = []
startChan  = 1

shift = [0,0,0]

with open(fileName,'w') as outfile:
    f = open(inFile, 'r')

    channel = 0
    data = {}
    data['detector']    = 'TAPS'
    data['parameters']  = parameters
    data['calEq']       = equations
    data['references']  = references
    data['ignore']      = ignore
    data['startChan']   = startChan
    data['channels']    = []
    
    for line in f:
        if('#' in line[0]): continue

        if('Shift-Coord:' in line):
            shift[2] = float(line.split()[2])
        
        if('Element:' not in line): continue
        columns = line.split()

        datum = {}
        datum['channel']      = channel
        datum['adc']          = [int(columns[1])]
        datum['scale']        = [float(columns[5])]
        datum['offset']       = [float(columns[2])]
        datum['position']     = [shift[i]+float(value) for i, value in enumerate(columns[11:14])]
        datum['raw']          = [[],[],[]]

        data['channels'].append(datum)
        
        channel    += 1
    
    channel = 0
    f = open(inFile, 'r')
    for line in f:
        if('#' in line): continue
        if('Next-TAPS:' not in line): continue
        columns = line.split()
        #print(line)
        data['channels'][channel]['neighbours'] = [int(i) for i in columns[3:]]

        channel    += 1
        
    f = open(inFile, 'r')

    channel = 0
    for line in f:
        if('#' in line): continue
        if('TAPSSG:' not in line): continue
        columns = line.split()

        data['channels'][channel]['adc']          += [int(columns[1]),int(columns[6].split('M')[0])]
        data['channels'][channel]['scale']        += [float(columns[5]),float(columns[10])]
        data['channels'][channel]['offset']       += [float(columns[2]),float(columns[9])]
        
        channel    += 1
    
    #print(data)
    json.dump(data, outfile, indent=2)

