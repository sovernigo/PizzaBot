from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import LeaveOneOut
import csv



from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

entidades = {
  "numInt": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  "numStr": ["um", "uma", "dois", "duas", "tres", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez"],
  "prato": ["pizza", "pizzas"],
  "sabores": ["calabreza", "frango", "mussarela", "mozarela", "3 queijos", "4 queijos", "a moda", "da casa", "a moda da casa", "lombo", "baiana"],
  "bebidas": ["agua", "aguas", "água", "águas", "coca", "coca-cola", "cocas", "coca-colas", "fanta", "fantas", "suco", "sucos", "uva", "pessego", "pêssego", "sprite", "sprites"],
  "detalhe": ["com catupiry", "ao creme"],
  "tipoAgua": ["gas", "gaseificada", "normal", "natural"]
}

agua = ["agua", "aguas", "água", "águas"]
refri = ["coca", "coca-cola", "cocas", "fanta", "fantas"]
suco = ["suco", "sucos", "uva", "pessego", "pêssego"]
saborSuco = ["uva", "pessego", "pêssego"]
lata = refri + suco

valorPizzaEsp = 39.90
valorPizzaClass = 29.90
lata = 5.00
aguaGar = 3.50


def etiquetador(dicionario, frase):
  reconhecido = list()

  frase = frase.replace(",", "")
  palavras = frase.split(" ")

  for palavra in palavras:
    for key, value in dicionario.items():
      if (palavra in value):
        reconhecido.append(str(key + " " + palavra))

  return reconhecido

'''def GeraPedido(itens):

  while()
  if(itens in entidades["prato"]):'''


#x, y = getData()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, strip_accents='unicode')

data = csv.reader(open('intencoes.csv', 'r'))
corpus, y = [], []
for row in data:
  corpus.append(row[1])
  y.append(row[0])

x = vectorizer.fit_transform(corpus)

modelLogistic = LogisticRegression()

loo = LeaveOneOut()
loo.get_n_splits(x)
y_pred = []
a = np.array(y)
for train_index, test_index in loo.split(x):
  x_train, x_test = x[train_index], x[test_index]
  y_train, y_test = a[train_index], a[test_index]

  modelLogistic.fit(x_train, y_train)
  y_pred.extend(modelLogistic.predict(x_test))                   #fit no modelo

#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, strip_accents='unicode')

while(1):


  x_in = input()                            #recebe entrada
  inst = vectorizer.transform([x_in])
  y_pred = modelLogistic.predict(inst)            #classifica entrada
  #print(y_pred)

  if(y_pred == 'saudacao'):                 #responde de acordo com a predicao
    print("Boa noite, bem-vindo à nossa pizzaria")

  elif(y_pred == "cardapio"):
    print("Trabalhamos com os seguintes itens:\n")
    print(" Pizzas Tradicionais - R$29.99\n Calabreza\n Mussarela\n Frango\n")
    print(" Pizzas Especiais - R$39.90\n 3 Queijos\n 4 Queijos\n À moda da casa\n Lombo ao Creme\n Baiana\n")
    print(" Bebidas:")
    print(" Água sem Gás - R$3.50")
    print(" Água com Gás - R$3.50")
    print(" Fanta Laranja lata - R$5.00")
    print(" Coca-Cola lata - R$5.00")
    print(" Suco Del Valle Pêssego lata - R$5.00")
    print(" Suco Del Valle Uva lata - R$5.00")

  elif(y_pred == 'informacao'):
    print("Nosso horário de funcionamento é de Terça à Domingo, das 18 às 23h30")

  elif(y_pred == 'garçom'):
    print("Garcom indo ate sua mesa")

  elif(y_pred == 'conta'):
    print("Conta indo ate sua mesa")

  elif(y_pred == 'pedido'):
    entd = etiquetador(entidades, x_in)
    comanda = GeraPedido(entd)
    print("Pedido realizado!")

  elif(y_pred == "cancelar pedido"):
    comanda = None
    print("Pedido Cancelado!")
      break

  #adiciona no novo csv x_in, y_pred
  with open('intencoes.csv', 'a') as newFile:
    newFileWriter = csv.writer(newFile)
    newFileWriter.writerow([x_in, y_pred])

  #x, y = getData()
  data = csv.reader(open('intencoes.csv', 'r'))
  corpus, y = [], []
  for row in data:
    corpus.append(row[1])
    y.append(row[0])

  x = vectorizer.fit_transform(corpus)
  a = np.array(y)
  modelLogistic.fit(x, a)                 #fit no modelo com a entrada nova
