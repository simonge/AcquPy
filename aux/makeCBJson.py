import json
import math

inFile = '/home/simong/git/acqu/acqu_user/data_MC12/CB_NaI_RecPol_Sep08_Sergey.dat'
tag = 'Test'

#inFile = '/home/simong/acqu-ROOT6/acqu_user/data.2016.06/Detector-NaI.dat'
#tag = 'New'

fileName = '/home/simong/AcquPy/aux/CrystalBall'+tag+'.json'

parameters = ['Energy','Time']
equations  = ["[(float(y['raw'][1]-y['raw'][0]))*y['scale'][i]+y['offset'][i] for y in x]","[(y['raw']-y['offset'][i]-ref[0])*y['scale'][i] for y in x]"]
#equations  = ["[y['raw'][2]*y['scale'][i]+y['offset'][i] for y in x]","[(y['raw']-y['offset'][i]-ref[0])*y['scale'][i] for y in x]"]
references = [[],[2000]]
ignore     = [26,29,30,31,32,33,34,35,36,37,38,40,311,315,316,318,319,353,354,355,356,357,358,359,360,361,362,363,364,365,366,400,401,402,405,408,679,681,682,683,684,685,686,687,688,689,691,692]


with open(fileName,'w') as outfile:
    f = open(inFile, 'r')

    channel = 0
    data = {}
    data['detector']    = 'CB'
    data['parameters']  = parameters
    data['calEq']       = equations
    data['references']  = references
    data['ignore']      = ignore
    data['channels']    = []
    
    for line in f:
        if('#' in line): continue
        if('Element:' not in line): continue
        columns = line.split()

        datum = {}
        datum['channel']      = channel
        datum['adc']          = [int(columns[1].split('M')[0]),int(columns[6].split('M')[0])]
        datum['scale']        = [float(columns[5]),float(columns[10])]
        datum['offset']       = [float(columns[2]),float(columns[9])]
        #r     = math.sqrt(float(columns[11])**2+float(columns[12])**2+float(columns[13])**2)
        #theta = math.acos(float(columns[13])/r)
        #phi   = math.atan2(float(columns[12]),float(columns[11]))
        #datum['position']     = [theta,phi,r]
        datum['raw']          = [[],[]]
        datum['position']     = [float(i) for i in columns[11:14]]

        data['channels'].append(datum)
        
        channel    += 1
    
    channel = 0
    f = open(inFile, 'r')
    for line in f:
        if('#' in line): continue
        if('Next-Neighbour:' not in line): continue
        columns = line.split()
        #print(line)
        data['channels'][channel]['neighbours'] = [int(i) for i in columns[3:]]

        channel    += 1
        
    #print(data)
    json.dump(data, outfile, indent=2)
