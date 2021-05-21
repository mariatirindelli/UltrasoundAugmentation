import os
import sys
os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
sys.path.append(os.path.abspath('C:\\GitRepo\\pyfl'))

# in configuration copy:
# Environment Variables: PYTHONUNBUFFERED = 1;PYTHONPATH=C:\Program Files\ImFusion\ImFusion Suite\Suite\;
