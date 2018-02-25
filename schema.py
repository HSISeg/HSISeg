import sqlite3

conn = sqlite3.connect('nnPU.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS PUstats (ID INTEGER PRIMARY KEY AUTOINCREMENT, pos_class VARCHAR(10), neg_class VARCHAR(70), precision DOUBLE,recall DOUBLE,true_pos INT,true_neg INT,false_pos INT,false_neg INT, test_type varchar(10), exclude_class_indx varchar(50), no_train_pos_labelled INT, no_train_pos_unlabelled INT, no_train_neg_unlabelled INT, visual_result_filename varchar(100), creation_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE INDEX IF NOT EXISTS PUstatspos ON PUstats (pos_class)''')
c.execute('''CREATE TABLE IF NOT EXISTS PNstats (ID INTEGER PRIMARY KEY AUTOINCREMENT, pos_class VARCHAR(10), neg_class VARCHAR(10), precision DOUBLE,recall DOUBLE,true_pos INT,true_neg INT,false_pos INT,false_neg INT, exclude_class_indx varchar(50), no_train_pos INT, no_train_neg INT, visual_result_filename varchar(100), creation_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE INDEX IF NOT EXISTS PNstatspos ON PNstats (pos_class)''')
c.execute('''CREATE TABLE IF NOT EXISTS PatchClass (ID INTEGER PRIMARY KEY AUTOINCREMENT, class BIGINT, patch_row_start BIGINT, patch_row_end BIGINT, patch_col_start BIGINT, patch_col_end BIGINT, creation_time DATETIME DEFAULT CURRENT_TIMESTAMP)''')
c.execute('''CREATE INDEX IF NOT EXISTS PatchClassOnclass ON PatchClass (class)''')
conn.commit()
conn.close()
