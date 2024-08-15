import argparse
from WTdenoising import denoise
from WearPre import predict
from Optimization import optimize
'''

1. Denoised
python main.py --denoise True --denoise_csvpath “..\\file.csv”

2. Prediction
python main.py --predict True --predict_csvpath “..\\file.csv”

3. Optimization
python main.py --optimize True --optimize_csvpath “..\\file.csv” 

'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--denoise', type=bool, default=False)
    parser.add_argument('--denoise_csvpath', type=str, default=None)
    parser.add_argument('--predict', type=bool, default=False)
    parser.add_argument('--predict_csvpath', type=str, default=None)
    parser.add_argument('--optimize', type=bool, default=False)
    parser.add_argument('--optimize_csvpath', type=str, default=None)

    config = parser.parse_args()

    #################################################################################
    if config.denoise:
        # Denoised
        if config.denoise_csvpath is None:
            raise ValueError("Input csv file for denoising is not provided！")

        Xkeys = ['Burial depth', 'UCS', 'Poisson ratio', 'Soil density', 'Porosity ratio', 'Coefficient of earth pressure',
                 'Wear', 'Cohesive force', 'Friction angle', 'Modulus of compressibility', 'Restricted particle size',
                 'Thrust', 'Torque', 'Penetration', 'SE', 'Efficiency']
        denoise(csvPath=config.denoise_csvpath, X_keys=Xkeys)

    if config.predict:
        # Prediction
        if config.predict_csvpath is None:
            raise ValueError("Input csv file for predicting is not provided！")

        Xkeys = ['Burial depth', 'UCS', 'Poisson ratio', 'Soil density', 'Porosity ratio', 'Coefficient of earth pressure',
                 'Radius', 'Cohesive force', 'Friction angle', 'Modulus of compressibility', 'Restricted particle size',
                 'Thrust', 'Torque', 'Penetration']
        predict(csvPath=config.predict_csvpath, X_keys=Xkeys)

    if config.optimize:
        # Optimization
        if config.optimize_csvpath is None:
            raise ValueError("Input csv file for optimizing is not provided！")

        predict(csvPath=config.optimize_csvpath)
