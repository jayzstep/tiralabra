# Testausdokumentti
## Miten testata?

Automaattiset testit voi suorittaa `numnet/` -kansiosta komennolla
```bash
pytest
```
## Mitä on testattu?
### Koulutus
Verkon kouluttamisen osalta on testattu että biasit ja weightit muuttuvat kouluttaessa verkkoa. Lisäksi on testattu että kouluttaminen overfittaa pienellä datamäärällä. Edellä mainitut testit eivät takaa, että neuroverkko toimii oikein, mutta ainakin tiedetään, että **jotain** tapahtuu.
### Tunnistaminen
Numeroiden tunnistamisesta on testattu taas, että vastauksena tulee **jotakin**, sekä että tulostuu ValueError, jos input-data ei ole verkon olettamassa muodossa. Automaattisen testauksen lisäksi kouluttaessa evaluate-funktio printtaa testidatan osumatarkkuuden joka eepokilla.
