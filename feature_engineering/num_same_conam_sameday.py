import multiprocessing as mp
import pandas as pd
import numpy as np
import time

pd.options.mode.chained_assignment = None


def convert_time_to_string(time):
    s_time = str(int(time))
    l = len(s_time)
    i = 6 - l
    res = '0' * i + s_time
    return res


def change_time_to_sec(date):
    date = convert_time_to_string(date)
    return int(date[:2]) * 60 * 60 + int(date[2:4]) * 60 + int(date[4:])


N_CORES = mp.cpu_count()

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# train = pd.read_csv('/Users/bensonbair/Documents/projects/tbrain_esbank_fraud_detection/data/train.csv')
# test = pd.read_csv('/Users/bensonbair/Documents/projects/tbrain_esbank_fraud_detection/data/test.csv')

# 將時間轉為秒
train['sec_time'] = train['loctm'].apply(lambda x: change_time_to_sec(x))
test['sec_time'] = test['loctm'].apply(lambda x: change_time_to_sec(x))

temp_df = pd.concat([train[['cano', 'conam', 'sec_time', 'locdt', 'txkey']],
                     test[['cano', 'conam', 'sec_time', 'locdt', 'txkey']]])
temp_df = temp_df.sort_values(['cano', 'locdt', 'sec_time']).reset_index(drop=True)


def time_interval_same_conam(arg):
    grp, lst = arg
    if (grp % 10000) == 0:
        print(grp)
    res = temp_df.loc[temp_df['cano'] == grp, :].reset_index(drop=True)

    res['num_same_conam_sameday'] = 0

    for i in range(len(res)):
        conam, day, second_th = res.loc[i, ['conam', 'locdt', 'second_th']]
        res2 = res.loc[(res['locdt'] == day)]

        if len(res2) > 1:
            res.loc[i, 'num_same_conam_sameday'] = np.sum(res2.conam == conam) - 1

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

    results_df = results_df.loc[:, ['txkey', 'num_same_conam_sameday']]

    results_df.to_csv('num_same_conam_sameday.csv', index=False)


if __name__ == '__main__':
    main()
