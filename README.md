# Tiralabra

NumNet on neuroverkko, joka tunnistaa käsin piirrettyjä numeroita MNIST-tietokannan datasta noin 95% tarkkuudella.

Mukana myös localhostiin käynnistyvä GUI, johon voi itse piirrellä numeroita. Osumatarkkuus on vähän heikompi, mutta edelleen ihan suht ok, riippuu käyttäjästä.

Lyhyesti: Lataa repo, lataa koulutusdata, anna Poetryn laulaa, fool around.

## Käyttöohje

### Kopioi repositio

```bash
git clone https://github.com/jayzstep/tiralabra.git
cd NumNet
mkdir data
```
### Lataa koulutusdata, eli MNIST-tietokanta

Lataa [täältä](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data) molemmat CSV-tiedostot ja laita ne kansioon `NumNet/data/`

### Asenna riippuvuudet (tarvitset Poetryn)

```bash
poetry install
```

### Käynnistä shell
```bash
poetry shell
```

### Treenaa verkko (jos haluat)
```bash
cd numnet
python3 nn.py train
```
Tämä myös käynnistää GUI:n localhostiin. Seuraa ruudun ohjeita.

**TAI**

### Käynnistä äppi suoraan
Treenivaiheen voi skipata ja serverin käynnistää suoraan komennolla:
```bash
python3 nn.py run
```
Gradio käynnistyy localhostiin, seuraa ruudun ohjeita.

## Testaus
Testit voi suorittaa `NumNet` -kansiosta käsin komennolla:
```bash
pytest
```
