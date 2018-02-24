import sqlite3

conn = sqlite3.connect('nnPU.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS PUstats (ID INTEGER PRIMARY KEY AUTOINCREMENT, pos_class VARCHAR(10), neg_class VARCHAR(70), precision DOUBLE,recall DOUBLE,true_pos BIGINT,true_neg BIGINT,false_pos BIGINT,false_neg BIGINT, test_type varchar(10), exclude_class_indx varchar(50), creation_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE INDEX IF NOT EXISTS PUstatspos ON PUstats (pos_class)''')
c.execute('''CREATE TABLE IF NOT EXISTS PNstats (ID INTEGER PRIMARY KEY AUTOINCREMENT, pos_class VARCHAR(10), neg_class VARCHAR(10), precision DOUBLE,recall DOUBLE,true_pos BIGINT,true_neg BIGINT,false_pos BIGINT,false_neg BIGINT,creation_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE INDEX IF NOT EXISTS PNstatspos ON PNstats (pos_class)''')
c.execute('''CREATE TABLE IF NOT EXISTS PatchClass (ID INTEGER PRIMARY KEY AUTOINCREMENT, class BIGINT, patch_row_start BIGINT, patch_row_end BIGINT, patch_col_start BIGINT, patch_col_end BIGINT, creation_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE INDEX IF NOT EXISTS PatchClassOnclass ON PatchClass (class)''')
conn.commit()
conn.close()
