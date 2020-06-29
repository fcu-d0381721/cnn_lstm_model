import numpy as np
from datetime import datetime

u=datetime.now()
t= np.array([x.strftime('%Y-%m-%d') for x in u], dtype='datetime64[h]')
# y = np.arange('2018-01-01', '2018-02-01', dtype='datetime64[h]')
print(t)
