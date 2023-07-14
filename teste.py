import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.impute import SimpleImputer


dados = pd.read_excel('C:\\Users\\desktop\\PycharmProjects\\IEEE_CIS\\Titanic.xlsx')
dados = dados.dropna(axis=1, how='all')

print('oi')

idades_filtradas = dados.dropna(subset=['Age']).copy()
imputer = SimpleImputer(strategy='mean')
dados_preenchidos = imputer.fit_transform(idades_filtradas[['Age']])

idades_filtradas.loc[:, 'Age_imputed'] = dados_preenchidos

plt.hist(idades_filtradas['Age_imputed'], bins=10, edgecolor='blue')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.title('Distribuição de Idades')
plt.show()

taxa_sobrevivencia = dados.dropna(subset=['Survived'])
labels = ['Sobreviveu', 'Não sobreviveu']
values = taxa_sobrevivencia['Survived'].value_counts()
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['blue', 'yellow'])

plt.title('Taxa de Sobrevivência')
plt.show()

X = idades_filtradas[['Age']]
y = idades_filtradas['Survived']

regressor = linear_model.LinearRegression()
regressor.fit(X, y)

plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, regressor.predict(X), color='red')
plt.xlabel('Idade')
plt.ylabel('Sobrevivência')
plt.title('Regressão Linear: Idade x Sobrevivência')
plt.show()

irmaos_a_bordo = dados.dropna(subset=['SibSp'])
irmaos_a_bordo = irmaos_a_bordo[irmaos_a_bordo['SibSp'] != 0]

plt.hist(irmaos_a_bordo['SibSp'], bins=10, color='red', edgecolor='black')
plt.xlabel('Quantidade de irmãos/cônjuges a bordo')
plt.ylabel('Frequência')
plt.title('Quantidade de irmãos/cônjuges a bordo')

plt.show()

sexo_passageiro = dados.dropna(subset=['Sex'])
labels = ['Homem', 'Mulher']
values = sexo_passageiro['Sex'].value_counts()
plt.pie(values, labels=labels, autopct='%.1f%%', colors=['orange', 'green'])

plt.title('Sexo dos passageiros')
plt.show()

embarque_encoded = pd.get_dummies(dados['Embarked'], prefix='Embarked')
dados_encoded = pd.concat([dados, embarque_encoded], axis=1)

embarque = dados.dropna(subset=['Embarked'])
labels = ['S = Southampton','C = Cherbourg', 'Q = Queenstown']
values = embarque['Embarked'].value_counts()

plt.pie(values, labels=labels, autopct='%.1f%%', colors=['blue', 'white', 'grey'])
plt.show()

pais_a_bordo=dados.dropna(subset=['Parch'])
pais_a_bordo=pais_a_bordo[pais_a_bordo['Parch']!=0]
plt.hist(pais_a_bordo['Parch'], bins=15, color='orange')
plt.xlabel('Pais/Filhos a bordo')
plt.ylabel('Frequência')
plt.title('Pais/Filhos a bordo')

plt.show()

classe_passageiro = dados.dropna(subset=['Pclass'])
labels = ['3','1','2']
values=classe_passageiro['Pclass'].value_counts()
plt.pie(values, labels=labels,autopct='%.1f%%', colors= ['red', 'green', 'grey'])
plt.show()