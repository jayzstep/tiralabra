# Tiralabra

NumNet on neuroverkko, joka tunnistaa käsin piirrettyjä numeroita MNIST-tietokannan datasta noin 95% tarkkuudella.

Mukana myös localhostiin käynnistyvä GUI, johon voi itse piirrellä numeroita. Osumatarkkuus on vähän heikompi, mutta edelleen ihan ok.

## Käyttöohje

### Kopioi repositio

```bash
git clone https://github.com/jayzstep/tiralabra.git
cd NumNet
mkdir data
```
### Lataa MNIST-tietokanta CSV:t

Lataa [täältä](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data) molemmat CSV-tiedostot ja laita ne kansioon `NumNet/data/`

### Asenna riippuvuudet

```bash
poetry install
```

### Käynnistä shell
```bash
poetry shell
```

### Treenaa verkko ja käynnistä äppi
```bash
cd numnet
python3 nn.py train
```
Gradio käynnistyy localhostiin, seuraa ruudun ohjeita.

### Käynnistä äppi
Olettaen että verkko on kerran koulutettu, voi treenivaiheen skipata ja serverin käynnistää suoraan komennolla:
```bash
python3 nn.py run
```

## Testaus
Testit voi suorittaa **projektin juuresta** komennolla:
```bash
pytest
```
