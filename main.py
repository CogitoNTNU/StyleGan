import logging
import os

# Silence Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import models.generator

generator = models.generator.SimpleGenerator()
generator.summary()