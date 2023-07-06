import pandas as pd
#import matplotlib as mtp

imdb_df = pd.read_csv("imdb_clean.csv")

imdb_df.columns = ["titulo", "ano de lancamento", "duracao", "genero", "nota", "Gross(M)"]