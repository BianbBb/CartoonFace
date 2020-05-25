# Task B: Recognition
'''
1. clip by box  and  resize
2. ResNeSt (+ FPN)  get feature(n)
3. NUT Module  n->1024
4. Circle loss


'''

import logging


LOG_FORMAT = "%(asctime)s[%(levelname)s]     %(message)s "
DATE_FORMAT = '%m-%d %H:%M:%S'
logging.basicConfig(level=logging.DEBUG,format=LOG_FORMAT, datefmt=DATE_FORMAT, )
logging.debug('----------  START  ---------')