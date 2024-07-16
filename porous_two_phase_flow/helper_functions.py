import numpy as np


def make_urf_array(urf, iter_max):
    if isinstance(urf, (list, tuple)):
        urf_array = np.asarray(urf)
        n_start = 0
        n_end = int(urf_array[0, 0])
        # urf_id_list = [list(range(n_end))]
        urf_value_list = [np.ones(n_end - n_start) * urf_array[1, 0]]
        for i in range(urf_array.shape[-1] - 1):
            n_start = int(urf_array[0, i])
            n_end = int(urf_array[0, i+1])
            # urf_id_list.append(list(range(n_start, n_end)))
            urf_value_list.append(np.ones(n_end - n_start) * urf_array[1, i+1])

        # urf_ids = [n for m in urf_id_list for n in m]
        urf_array = np.concatenate(urf_value_list, axis=0)

        if iter_max > n_end:
            n_ext = iter_max - n_end
            urf_array = np.concatenate(
                (urf_array, np.ones(n_ext) * urf_array[-1]), axis=0)
        # urf_list = urf_values]
    elif isinstance(urf, float):
        urf_array = np.ones(iter_max) * urf
    else:
        urf_array = None
    return urf_array
