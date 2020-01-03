import os
import datetime
import numpy as np

END_DATE = '2020-01-01'
DATE_FORMAT = '%Y-%m-%d'
EPS = np.finfo(np.float32).eps
MULTIPLIER = 1    #   1000
PERCISION = 4
CUR_DIR = os.path.dirname(__file__)
MAIN_DIR = os.path.dirname(CUR_DIR)
DATASETS_DIR = MAIN_DIR + '/datasets/'
OUTPUTS_DIR = MAIN_DIR + '/outputs/'
MODELS_DIR = MAIN_DIR + '/models/'
CONFIG_DIR = MAIN_DIR + '/config'
sample_start_date = '2010-01-01'
sample_start_dateTIME = datetime.datetime.strptime(sample_start_date, DATE_FORMAT)
CAPITAL_BASE_MULTIPLIER = 0.3
MAX_WEIGHT = 1.0
RISK = 0.2