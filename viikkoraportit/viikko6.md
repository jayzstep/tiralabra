## Tiistai 3h

Päivittelin vähän kommentteja vertaisarvioijaa varten. W ja B testit olivat aika tyhmät, joten päivitin ne käyttämään numpyn array_equalia.
Hakkasin päätä seinään testin kanssa, joka testaa onko samplejen järjestyksellä väliä. Mutta nyt tiedän mitä pitää tehdä. Huomisen hommia.

## Keskiviikko 3h

Refaktoroin forwardpropagationin omaksi funktioksi, niin pääsee sitä testaamaan erikseen. Tein myös sille samplejärjestys testin. Huomasin myös, että 
testatessa voi olla mukava, jos käyttää seediä arpoessa w ja b arvoja, joten säädin vähän verkkoluokkaa ja testejä, niin että arvotut painot ja biasit
pysyy samoina. En koskaan huomannut siinä mitään ongelmaa, mutta testaamismatskujen artikkelissa oli mainittu että seed voi olla hyvä idea.

## Torstai 3.5h
Säädin vähän hidden layerin kokoa, sain osumatarkkuuden 97 prosenttiin! Lisäksi tein vertaisarvioita.

Palautteen perustella eriytetty verkko ja sen funktiot omaksi fileeksi ja lisäsin main.py:n jossa UI ym. En vieläkään tiedä onko sigmoidia ym funktioita
tarpeellista testata? Ehkä more is more.

