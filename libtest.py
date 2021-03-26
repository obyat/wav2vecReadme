from __future__ import absolute_import, division, print_function, unicode_literals
print("importing...")
import argparse
import concurrent.futures
import json
import multiprocessing
import importlib
import os

from collections import namedtuple
from itertools import chain

import sentencepiece as spm


import torch as t
import transformers
import tqdm

import os
from tqdm import tqdm

import numpy as np




import numpy as np
import core
import transformers


import numpy as np
import os
import json



import os
from tqdm import tqdm
from core import load_file
import re
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations_with_replacement
import jiwer
from collections import Counter
import math


import os
from tqdm import tqdm
from core import load_file


import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import itertools as it
import re
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import warnings
from transformers import BertLayer, BertModel, BertForMaskedLM, BertForPreTraining
from transformers.models.bert.modeling_bert import (
    BertForPreTrainingOutput,
    BertPreTrainingHeads,
    BertEncoder,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
#including core.py



from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import core
import os

import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json




import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import torch
import argparse
import librosa
import torch.nn.functional as F
import itertools as it
'''
NOT WORKING
from fairseq import utils
from fairseq.models import BaseFairseqModel
from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
from fairseq.data import data_utils
from fairseq.models.wav2vec.wav2vec2_asr import base_architecture, Wav2VecEncoder
from wav2letter.decoder import CriterionType
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from fairseq import utils
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
'''
import re
import numpy as np

from collections import Counter
from multiprocessing import Pool

import torch

import contextlib





"""
#working!

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from torch.nn.utils.rnn import pad_sequence
import torch
import core




import transformers
import re
import numpy as np
import itertools as it
import core


import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm
from torchvision import datasets, models, transforms
import time
import copy
from transformers import AdamW
from torch import nn
import re
import transformers
import torch.nn.functional as F
from torch.optim import RMSprop
import itertools as it
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error
from datetime import datetime
import json
import argparse
import gc

import core
import dataloaders


import torch
import argparse
import soundfile as sf
import torch.nn.functional as F
import itertools as it

import os

use_gpu_num = '-1'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data.dataset import Dataset
import time
import copy
from transformers import AdamW
from torch.optim import RMSprop
import argparse
import soundfile as sf
import librosa
import torch.nn.functional as F
import itertools as it

import torch.nn as nn

import numpy as np
from tqdm import tqdm
import contextlib
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import gc
import itertools as it
import os.path as osp
import warnings
from collections import deque, namedtuple

import numpy as np
import torch
"""



import logging
import math
import os
import sys

import editdistance
import numpy as np
import torch
import importlib

import logging
import math

import torch
import torch.nn.functional as F











import argparse
import math
from collections.abc import Iterable

import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re
import sys
import re
from collections import deque
from enum import Enum

import numpy as np

import torch

#from examples.speech_recognition import criterions, models, tasks 
#from examples.speech_recognition.data import AsrDataset
#from examples.speech_recognition.data.replabels import replabel_symbol
#from examples.speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
print("imports complete")
