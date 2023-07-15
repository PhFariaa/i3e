import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import time

imdb_df = pd.read_csv("imdb_clean.csv")

imdb_df.columns = ["titulo", "ano de lancamento", "duracao", "genero", "notas", "Money(M)"]


def calcula_media_por_genero(imdb_df, genero):
    imdb_df_filter  = (imdb_df['genero'] ==genero) & (imdb_df['notas'])
    return imdb_df.loc[imdb_df_filter, ['notas']].mean()
    

def calcula_media_por_dinheiro_gasto(imdb_df, genero):
    imdb_df_filter  = (imdb_df['genero'] ==genero) & (imdb_df['Money(M)'])
    return int(imdb_df.loc[imdb_df_filter, ['Money(M)']].mean())

generos = imdb_df['genero'].str.split().explode().unique()

for genero in generos:
    filme_genero_nota = calcula_media_por_genero(imdb_df, genero)
    #print(f"A media de notas dos filmes do genero {genero} é {filme_genero_nota}")

media_notas_df = (imdb_df.groupby('genero')[['notas']].mean()) #media de notas de todos os generos
#print(media_notas_df)
maior_media = media_notas_df.max()#maior media entre os generos

lista_gasto = []
maior_gasto = 0
i = 0

media_dinheiro_gasto_df = (imdb_df.groupby('genero')[['Money(M)']].mean())
    
maior_gasto = media_dinheiro_gasto_df.max() #genero que mais gasta

#pegar os 5 generos que mais gastam e colocar eles em um grafico de comparacao
eixo_y = [146.1, 132.8, 121.6, 119, 100.6]
eixo_x = ['aventura', 'sci-fi', 'fantasia', 'acao', 'animacao']
plt.bar(eixo_x, eixo_y)
plt.ylabel('Dinheiro gasto em Milhoes')
plt.xlabel('Generos')
plt.title('Dinheiro gasto por genero de filme')
#plt.show()

dataframe_dados_num = imdb_df.copy()
dataframe_dados_num = dataframe_dados_num.drop('titulo', axis=1)
dataframe_dados_num = dataframe_dados_num.drop('genero', axis=1)
dataframe_dados_num = dataframe_dados_num.drop('ano de lancamento', axis=1)

X = dataframe_dados_num.drop('notas', axis=1)
y = imdb_df["notas"].values

from sklearn.model_selection import train_test_split

XTrain, XTeste, yTrain, yTest = train_test_split(X, y, test_size = 0.15)

from sklearn.preprocessing import StandardScaler

scale_obj = StandardScaler()
X = scale_obj.fit_transform(X.astype(float))

from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
XTrain = pca.fit_transform(XTrain)
XTeste = pca.transform(XTeste)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver= 'lbfgs')
start_time = time.perf_counter()
model.fit(XTrain, yTrain)
end_time = time.perf_counter()
print(end_time - start_time)

regressor = linear_model.LinearRegression()
regressor.fit(XTrain, yTrain)
yPred = regressor.predict(XTeste, yTest)
