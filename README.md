To run it create a virtualenv with python 3 and activate it.
Run the following commands
1) pip3 install -r requirement.txt
2) python3 manage.py migrate


Now open the following files
1) Config.py comment line 1-6 and uncomment line 7-11

run the experiment with 

1) python3 run_exp.py --data Indian_pines
2) python3 run_exp.py --data Salinas
3) python3 run_exp.py --data PaviaU
4) python3 run_PN_multi_exp.py --data Indian_pines
5) python3 run_PN_multi_exp.py --data Salinas
6) python3 run_PN_multi_exp.py --data PaviaU
7) python3 run_PN_exp.py --data Indian_pines
8) python3 run_PN_exp.py --data Salinas
9) python3 run_PN_exp.py --data PaviaU