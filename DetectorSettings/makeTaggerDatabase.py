import sqlite3

con = sqlite3.connect('/home/simong/AcquPy/aux/db.sqlite3')
cur = con.cursor()

cur.execute("DROP TABLE IF EXISTS tagger")

tagger_sql = """
CREATE TABLE tagger (
id           integer PRIMARY KEY,
channeladc   integer NOT NULL,
referenceadc integer NOT NULL,  
channel      integer NOT NULL,  
energy       real    NOT NULL,
ewidth       real    NOT NULL,
scale        real    NOT NULL,
shift        real    NOT NULL
)"""

cur.execute(tagger_sql)

element_sql = "INSERT INTO tagger (channeladc, referenceadc, channel, energy, ewidth, scale, shift) VALUES (?, ?, ?, ?, ?, ?, ?)"
f = open('/home/simong/git/acqu/acqu_user/data_MC12/FP_1508.RecPol.09.08.dat', 'r')

referenceADC = 1400
channel = 0

for line in f:
    if('#' in line): continue
    if('Element:' not in line): continue
    columns = line.split()
    
    channeladc = columns[6].split('M')[0]
    energy     = columns[14]
    ewidth     = columns[15]
    scale      = columns[10]
    shift      = columns[9]

    print channeladc, referenceADC, channel, energy, ewidth, scale, shift

    cur.execute(element_sql, (channeladc, referenceADC, channel, energy, ewidth, scale, shift))    

    channel    += 1

con.commit()
