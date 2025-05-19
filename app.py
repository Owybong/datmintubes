import pandas as pd

dt = pd.read_csv('listings.csv')

df = pd.DataFrame(dt)

df.info()