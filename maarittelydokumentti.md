# Määrittelydokumentti

Opinto-ohjelma: tietojenkäsittelytieteen kandidaatti (TKT)

Projekti toteutetaan Pythonilla. Voin vertaisarvioida muita Python-projekteja.

Toteutan neuroverkon vastavirta-algoritmeineen, joka ohjatusti oppii tunnistamaan käsinkirjoitettuja numeroita. Käytän treeni- ja testausdatana MNIST-tietokantaa. Matriisilaskennassa apuna käytän Numpy-kirjastoa. 

Valmis ohjelma näyttää muutaman (5-10) satunnaisen kuvan datasta sekä esittää oman arvionsa siitä, mikä luku on kyseessä. 

Aika- ja tilavaativuuden osalta tavoitteena on olla linjassa muiden vastaavien toteutusten kanssa. Yksittäisen kuvan tunnistamiseen verkkoa tarvitsee vain yhteen suuntaan. Aikavaatimus on vähintään samaa luokkaa matriisin kertolaskun kanssa, joiden määrä riippuu verkon kerrosten määrästä [1].

[1] https://lunalux.io/computational-complexity-of-neural-networks/
