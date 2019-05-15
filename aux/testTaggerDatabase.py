import sqlite3

con = sqlite3.connect('/home/simong/AcquPy/aux/db.sqlite3')
cur = con.cursor()

cur.execute("SELECT channel, energy, channeladc FROM tagger")
results = cur.fetchall()
print results
