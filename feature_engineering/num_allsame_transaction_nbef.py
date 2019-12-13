import multiprocessing as mp
import pandas as pd
import numpy as np
import time

pd.options.mode.chained_assignment = None

N_CORES = mp.cpu_count()

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


temp_df = pd.concat([train[['cano', 'acqic', 'contp', 'csmcu', 'etymd', 'hcefg', 'conam', 'mcc', 'mchno', 'locdt', 'loctm', 'txkey']],
                     test[['cano', 'acqic', 'contp', 'csmcu', 'etymd', 'hcefg', 'conam', 'mcc', 'mchno', 'locdt', 'loctm', 'txkey']]])
temp_df = temp_df.sort_values(['cano', 'locdt', 'loctm']).reset_index(drop=True)


def time_interval_same_conam(arg):
    grp, lst = arg
    if (grp % 10000) == 0:
        print(grp)
    res = temp_df.loc[temp_df['cano'] == grp, :].reset_index(drop=True)
    res['num_allsame_transaction'] = 0
    res['avg_num_allsame_transaction_bef'] = 0

    for i in range(len(res)):
        locdt, acqic, contp, csmcu, etymd, hcefg, mcc, mchno = res.loc[i, ['locdt', 'acqic', 'contp', 'csmcu', 'etymd', 'hcefg', 'mcc', 'mchno']].values

        res2 = res.loc[(res['locdt'] == locdt) &
                       (res['acqic'] == acqic) &
                       (res['contp'] == contp) &
                       (res['csmcu'] == csmcu) &
                       (res['etymd'] == etymd) &
                       (res['hcefg'] == hcefg) &
                       (res['mcc'] == mcc) &
                       (res['mchno'] == mchno)]

        res3_all = res.loc[(res['locdt'] < locdt)]
        # print(res3_all)

        n2 = len(res2)

        res.loc[i, 'num_allsame_transaction'] = n2
        # print(res)
        if len(res3_all) != 0:
            l = len(set(res3_all.locdt))
            # print(i, l)
            res3 = res3_all.loc[(res3_all['acqic'] == acqic) &
                                (res3_all['contp'] == contp) &
                                (res3_all['csmcu'] == csmcu) &
                                (res3_all['etymd'] == etymd) &
                                (res3_all['hcefg'] == hcefg) &
                                (res3_all['mcc'] == mcc) &
                                (res3_all['mchno'] == mchno)]
            # print(i, 'res3', res3)
            n3 = len(res3)
            if n3 > 0:
                res.loc[i, 'avg_num_allsame_transaction_bef'] = n3 / l

    res = res.reset_index()
    # print(res)
    return res


def main():
    global temp_df

    grp_lst_args = list(temp_df.groupby('cano').groups.items())

    start_time = time.time()

    pool = mp.Pool(processes=N_CORES)
    results = pool.map(time_interval_same_conam, grp_lst_args)
    pool.close()
    pool.join()

    results_df = pd.concat(results)

    print("--- %s seconds ---" % (time.time() - start_time))

    results_df = results_df.reset_index(drop=True)

    results_df = results_df.loc[:, ['txkey', 'num_allsame_transaction', 'avg_num_allsame_transaction_bef']]

    results_df.to_csv('num_allsame_transaction_nbef.csv', index=False)


if __name__ == '__main__':
    main()
