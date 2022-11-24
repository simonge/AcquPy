f = open('/home/simong/git/acqu/acqu_user/data_MC12/FP_1508.RecPol.09.08.dat', 'r')

adc = []

for line in f:
    if('#' in line): continue
    if('Element:' not in line): continue
    columns = line.split()
    value = columns[6].split('M')
    adc.append(int(value[0]))

print adc
