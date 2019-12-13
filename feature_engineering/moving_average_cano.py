import multiprocessing as mp
import pandas as pd
import numpy as np
import time

pd.options.mode.chained_assignment = None

N_CORES = mp.cpu_count()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

temp_df = pd.concat([train[['cano', 'locdt', 'conam']], test[['cano', 'locdt', 'conam']]]).reset_index()
temp_df = temp_df.groupby(['cano', 'locdt'])['conam'].agg(['mean'])
temp_df = temp_df.reset_index()


def compute_moving_average_3_noshift(arg):

    grp, lst = arg
    res = temp_df.loc[temp_df['cano'] == grp, :]
    res['cano_locdt_move_average_3day_noshift'] = res['mean'].rolling(window=3).mean()
    res = res.reset_index()
    # print(res)
    return res


def main():
    global temp_df

    start_time = time.time()

    grp_lst_args = list(temp_df.groupby('cano').groups.items())

    pool = mp.Pool(processes=N_CORES)
    results = pool.map(compute_moving_average_3_noshift, grp_lst_args)
    pool.close()
    pool.join()

    results_df = pd.concat(results)
    results_df.to_csv('cano_locdt_move_average_3day_noshift.csv', index = False)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
