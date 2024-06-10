# Tiralabra

**HUOM**
Huomasin juuri että GUI ei toimi oikein Chromella. Safarilla toimii niin kuin pitää, Firefoxista en tiedä. Chromella ei pysty raahaamaan testikuvia, vaan pitää klikata "Upload" ja manuaalisesti valita kuva `data/testSample` -kansiosta.
Pahoittelut tästä, rakas vertaisarvioija. Koitan selvittää mistä kyse.
**huom**

NumNet on neuroverkko, joka tunnistaa käsin piirrettyjä numeroita MNIST-tietokannan datasta noin 95% tarkkuudella.

Mukana myös localhostiin käynnistyvä GUI, jossa voi raahata samplekuvia verkolle tunnistettavaksi.

Voit itse valita koulutatko verkon uudestaan, vai käytätkö valmiiksi koulutettua verkkoa komennoilla "train" tai "run". Tarkemmat ohjeet alla.
Koulutuksessa menee 1-10 minuuttia, riippuu koneen spekseistä.

Lyhyesti: Lataa repo, lataa koulutusdata ja samplet, anna Poetryn laulaa, fool around.

## Käyttöohje

### Kopioi repositio

```bash
git clone https://github.com/jayzstep/tiralabra.git
cd NumNet
mkdir data
```
### Lataa koulutusdata, eli MNIST-tietokanta

Lataa [täältä](https://www.dropbox.com/scl/fi/t3z7uidb1q5myqrhfbnb1/Arkisto.zip?rlkey=27iowi7reqz9khpylg8a6xrhl&st=ki8kxbsg&dl=0) molemmat CSV-tiedostot sekä samplekuvat ja laita/pura ne tässä muodossa kansioon `NumNet/data/`

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
