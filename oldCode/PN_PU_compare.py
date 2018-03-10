from  create_binary_data import save_data
from PN_learning import run_classification
from train import main
import csv

n_class = 9
file_path = "PN_PU_res_random_neg.csv"
def run_PN_PU_classification(pos_class, neg_class):
    save_data(pos_class, neg_class, True)
    precisionPN, recallPN, (tnPN, fpPN, fnPN, tpPN) = run_classification()
    precisionPU, recallPU, (tnPU, fpPU, fnPU, tpPU) = main()
    return precisionPN, recallPN, (tnPN, fpPN, fnPN, tpPN), precisionPU, recallPU, (tnPU, fpPU, fnPU, tpPU)

def save_data_in_file(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(data)


def main_compare():
    headers = ['(Pos)', 'precisionPN', 'recallPN', 'precisionPU', 'recallPU', 'tnPN', 'fpPN', 'fnPN', 'tpPN', 'tnPU', 'fpPU', 'fnPU', 'tpPU' ]
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(headers)

    # for i in range(0, n_class - 1):
    #     for j in range(i+1, n_class):
    #         pos_class = i
    #         neg_class = j
    #         precisionPN, recallPN, (tnPN, fpPN, fnPN, tpPN), precisionPU, recallPU, (tnPU, fpPU, fnPU, tpPU) = run_PN_PU_classification(pos_class, neg_class)
    #         data = [str((pos_class,neg_class)), str(precisionPN), str(recallPN), str(precisionPU), str(recallPU), str(tnPN), str(fpPN), str(fnPN), str(tpPN), str(tnPU), str(fpPU), str(fnPU), str(tpPU)]
    #         save_data_in_file(file_path, data)
    #         print((i, j), " done")
    #         pos_class = j
    #         neg_class = i
    #         precisionPN, recallPN, (tnPN, fpPN, fnPN, tpPN), precisionPU, recallPU, (tnPU, fpPU, fnPU, tpPU) = run_PN_PU_classification(pos_class, neg_class)
    #         data = [str((pos_class, neg_class)), str(precisionPN), str(recallPN), str(precisionPU), str(recallPU),
    #                 str(tnPN), str(fpPN), str(fnPN), str(tpPN), str(tnPU), str(fpPU), str(fnPU), str(tpPU)]
    #         save_data_in_file(file_path, data)
    #         print((j, i), " done")

    for i in range(0, n_class):
        j = i + 1
        pos_class = i
        neg_class = j
        precisionPN, recallPN, (tnPN, fpPN, fnPN, tpPN), precisionPU, recallPU, (tnPU, fpPU, fnPU, tpPU) = run_PN_PU_classification(pos_class, neg_class)
        data = [str(pos_class), str(precisionPN), str(recallPN), str(precisionPU), str(recallPU), str(tnPN), str(fpPN), str(fnPN), str(tpPN), str(tnPU), str(fpPU), str(fnPU), str(tpPU)]
        save_data_in_file(file_path, data)
        print(i, " done")

if __name__ == '__main__':
    main_compare()