import pandas as pd
import numpy as np
import pywt
from os import path

def denoise(csvPath, X_keys):
    '''
    Args:
    csvPath : Absolute path to import csv file
    
    '''

    # Read data
    data = pd.read_csv(csvPath)


    columns_to_denoise = X_keys
    best_snrs = dict()

    for i in columns_to_denoise:
        data_to_denoise = data[i]
        data2 = np.array(data_to_denoise)
        best_snr_func = 0
        threshold = np.std(data2) * np.sqrt(2 * np.log(len(data2)))
        # Selection of wavelet basis functions
        for j in range(1, 11):
            coeffs = pywt.wavedec(data2, 'db{}'.format(j), level=6, mode='symmetric')
            coeffs[1:] = [pywt.threshold(k, threshold, 'hard') for k in coeffs[1:]]
            data_denoised = pywt.waverec(coeffs, 'db{}'.format(j), mode='symmetric')
            if len(data_denoised) != len(data2):
                data_denoised = data_denoised[:-1]
            snr = 10 * np.log10(np.sum(data2 ** 2) / np.sum((data2 - data_denoised) ** 2))
            if snr > best_snr_func:
            #if rmse < best_rmse:
                best_snr_func = snr
                #best_rmse = rmse
                best_db = "db{}".format(j)
            best_snrs[i] = [best_db]
            #best_snrs[i] = [best_db, best_rmse]
        # Select the number of decomposition layers
        best_snr = 0
        for t in range(2, 5):
            coeffs = pywt.wavedec(data2, best_snrs[i][0], level=t, mode='symmetric')
            coeffs[1:] = [pywt.threshold(k, threshold, 'hard') for k in coeffs[1:]]
            data_denoised = pywt.waverec(coeffs, best_snrs[i][0], mode='symmetric')
            if len(data_denoised) != len(data2):
                data_denoised = data_denoised[:-1]
            snr = 10 * np.log10(np.sum(data2 ** 2) / np.sum((data2 - data_denoised) ** 2))
            if snr > best_snr:
            #if rmse < best_rmse:
                best_snr = snr
                #best_rmse = rmse
                best_level = t
            best_snrs[i] = [best_db, best_snr, best_level]
            db = best_snrs[i][0]
            lev = best_snrs[i][2]
            threshold = np.std(data2) * np.sqrt(2 * np.log(len(data2)))
            coeffs = pywt.wavedec(data2, db, level=lev, mode='symmetric')
            coeffs[1:] = [pywt.threshold(i, threshold, 'hard') for i in coeffs[1:]]
            data_denoised = pywt.waverec(coeffs, db, mode='symmetric')
            if len(data_denoised) != len(data2):
                data_denoised = data_denoised[:-1]
            data = pd.DataFrame(data)
            data.loc[:, [i]] = data_denoised.reshape((len(data_denoised), 1))
    WTresult = pd.DataFrame(best_snrs)

    # Save the result file in the same directory as csvPath, with the filename data_denoised.csv.
    savePath = path.join(path.dirname(csvPath), 'data_denoisedthre.csv')
    data.to_csv(savePath, mode='w', index=False)
    WTresult_path = path.join(path.dirname(csvPath), 'WTresultthre.csv')
    WTresult.to_csv(WTresult_path, mode='w', index=False)
