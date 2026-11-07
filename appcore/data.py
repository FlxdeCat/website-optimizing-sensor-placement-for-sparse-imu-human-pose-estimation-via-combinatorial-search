import json
from functools import lru_cache
import pandas as pd
from .constants import ROOT
@lru_cache(maxsize=1)
def results():return json.loads((ROOT/'assets/results.json').read_text(encoding='utf-8'))
def rows():return pd.DataFrame(results()['rows'])
def sequence_rows():return pd.DataFrame(results()['per_sequence'])
def manifest():return json.loads((ROOT/'assets/models.json').read_text())
