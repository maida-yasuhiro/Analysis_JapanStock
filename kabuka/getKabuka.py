import pandas as pd
import csv
import shlex,subprocess

df = pd.read_csv('../meigara.csv')
df = pd.DataFrame(df)
df_lists = list(df.values.flatten())

for df_list in df_lists:
  subprocess.call("python jpstock.py %d 2015-01-30" % df_list, shell=True)
