import sys
import os


script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'common', 'mylog')
sys.path.append( mymodule_dir )
for p in sys.path:
    print( p )

# import common.mylog.mylog error
# from common.mylog import mylog error
import log


# ok !!!!!
# sys.path.append( '../common/mylog' )
# import log
# log.say_hello()


log.say_hello()
# logger.info("---test log-----------")
print("----end----")
log.loginfo("test log")

# okkkkkkk
from log import logger
logger.info("  logger ok!!!!!!!!!!!")