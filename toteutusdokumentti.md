# Toteutusdokumentti

## Ohjelman yleisrakenne

Ohjelma lataa CSV-tiedostosta MNIST-tietokantaan tallennetut 60 000 kuvaa käsin piirretyistä numeroista ja muuntaa 28x28 pikselin kuvat 784 pituisiksi vektoreiksi.
Tämän lisäksi ladataan myös 10 000 testaamiseen tarkoitettua kuvaa.

Neuroverkolle annetaan alkuarvoina verkon kerrosten koot. Verkko on yksinkertainen yhden piilokerroksen verkko, joten input-kerros on kokoa 784, piilokerroksella on 60 neuronia (tätä voi koodista muuttaa) ja output-kerros on 10 pituinen vektori, sillä numerot jaotellaan luonnollisesti 10 eri lokeroon. Annetun verkon koon perusteella ohjelma alustaa joka kerrokselle sopivat sattumanvaraiset painot (matriisi) ja bias (vektori), joita lähdetään sitten koulutusvaiheessa päivittämään kohti parempia arvoja. Verkolle määritellään myös laskennassa käytettävät aktivaatio- ja error-funktiot. Tämä kyseinen ohjelma käyttää aktivaationa Sigmoidia ja errorina MSE-funktiota(mean square error).

Verkko koulutetaan ajamalla koko koulutusdata verkon läpi useita kertoja, näitä kutsutaan eepokeiksi, ja päivittämällä verkon kerrosten paino-, ja bias-arvoja. Eepokkien määrää voi muuttaa koodissa. Tällä hetkellä eepokkien määrä on 30. Koulutus tapahtuu 10 samplen "paketeissa" (nk. mini_batch, tämäkin on koodissa muutettavissa). Koulutuksen voisi toteuttaa yksi sample kerrallaan, mutta tällöin jäisi hyödyntämättä Numpy-kirjaston varsin tehokkaat matriisien kertolaskualgoritmit. Koulutuksessa samplepaketti muutetaan matriisiksi, joka kerrotaan kerroksen painomatriiseilla ja joka sarakkeelle (eli joka samplelle) lisätään kerroksen bias-arvot, lopuksi joka arvo ajetaan aktivaatiofunktion (Sigmoid) läpi elementti kerrallaan. Sigmoid normalisoi arvot 0 ja 1 välille. Lopulta aktivoitu kerroksen output-matriisi toimii inputtina seuraavalle kerrokselle ja prosessi toistetaan.

Kun samplepaketti on ajettu verkon läpi, verrataan outputtina saatavia arvoja koulutusdatan mukana tuleviin y-arvoihin, jotka kertovat mitä niiden pitäisi olla, eli mitä kuvassa oikeasti oli. Tästä lasketaan virheen määrä error-funktiolla (mse). Tästä saadulle virhearvolle päädytään laskemaan derivoimalla miten verkon eri painot ja biasit vaikuttavat sen määrään. Kun jokaiselle painolle ja biasille on laskettu kulmakerroin (gradient), painoja ja biaseja päivitetään gradientin vastakkaiseen suuntaan pienen "nykäyksen" verran, nykäyksen kokoa kutsutaan oppimisnopeudeksi (learning rate). Näin verkko oppii.

Miksi piilokerroksen ja samplepakettien kokoa sekä oppimisnopeutta ja samplepakettien määrää voi muuttaa koodissa verkko-objektia luotaessa? Koska näiden "hyperparametrien" optimointi on olennainen osa verkon toimivuuden varmistamista. Eri arvoilla saadaan hyvinkin erilaisia oppimistuloksia. Tästä lisää puutteissa ja parannusehdotuksissa.

Verkkoa kouluttaessa ohjelma testaa verkon osumatarkkuutta jokaisen eepokin jälkeen erillisellä testidatalla. Lopuksi ohjelma printtaa kaavioon käppyrän koulutusprosessin kulusta. Ohjelma myös tallentaa koulutetut painot ja biasit tiedostoon, josta ne on helppo ladata jatkoa varten. Verkkoa ei siis ole tarpeen kouluttaa joka käyttökerralla.

Koulutuksen jälkeen localhostiin käynnistyy UI, jossa käyttäjä voi kuvia klikkaamalla (raahaus toimii vain Safarilla) testata verkon toimintaa.

## Aika-, tila- ja suorituskykyanalyysi

Suorituskyky ei ole tämän projektin simppelin neuroverkon kannalta kovinkaan olennainen hiottava parametri. Kun verkko on kerran koulutettu, ei sitä tarvitse kouluttaa uudestaan ja samplekuvan lajittelu on riittävän nopeaa. Kuvan datan vienti verkon läpi ja luokittelutuloksen saanti vaatii vain muutaman matriisikertolaskun. Koulutusta varten tämä ohjelma on toteutettu hyödyntäen Numpy-kirjaston hyvin optimoituja matriisilaskuja. Koulutuksen voisi toteuttaa myös for-loopilla sample kerrallaan, mutta tällöin matriisilaskun nopeuttavat hyödyt jäävät saamatta.

Yleisesti matriisikertolaskun aikavaatimus on $O(n^3)$ [2] ja elementtikohtaisen matriisilaskun (esim. aktivaatiofunktio) aikavaatimus on $O(n)$ [1]. Neuroverkossa eteenpäin mentäessä nämä lasketaan joka kerroksessa (poislukien input). Vastavirta koostuu samoista operaatioista ja vaatimus myöskin riippuu kerrosten määrästä. Lisäksi lineaarialgebraa käytetään matriisin transpoosin laskemiseen, jonka aikavaatimus on $O(mn)$ [3].

Luna Lux Fredenslund päätyy artikkelissaan [1] laskelmaan, jossa inputin vieminen neuroverkon läpi vie aikaa $O(n^4)$ ja vastavirta-algoritmin ollessa huomattavasti hitaampi $O(n^5)$.

## Puutteet ja parannusehdotukset

Suurin kysymys koko ohjelmalle on, olisiko pitänyt ratkaista projektiksi määritelty MNIST-jaottelu, vai toteuttaa kokonainen neuroverkkokirjasto. Tämän hetken toteutus on joiltain osin aika lähellä kirjastoa, mutta olennaisilta osin myös täysin räätälöity, yksittäinen ratkaisu MNIST-ongelmaan. Koodista saisi aika pienellä vaivalla kirjastomaisemman ja kyvykkään ratkaisemaan muitakin tunnettuja lajitteluongelmia. Suurin rajoite tällä hetkellä on "kovakoodattu" yhden piilokerroksen rajoitus. Muuntamalla vastavirta-algoritmin toteutusta hiukan, saisi piilokerroksen nabla-muuttujat ja painojen ja biasien päivitys toteutettua looppaamalla niin, että kerroksia voisi olla käyttäjän määrittelemä lukumäärä. Myös muuntamalla hieman isommin koodin rakennetta voisi joka kerrokselle määritellä omat aktivaatiofunktiot. Tämä vaatisi (esimerkiksi) jokaisen kerroksen toteuttamisen omana luokkanaan.

Ohjelmalla oli aluksi toteutettuna GUI, jossa käyttäjä pystyi piirtämään itse numeroita verkolle tunnistettavaksi, mutta verkko ei ollut kovinkaan hyvä tunnistamaan niitä (ehkä n. 50-60% meni oikein). Tästä kokeilusta jäi kuitenkin olo, että tällainen ominaisuus olisi ihan toteutettavissa, mutta se jäi nyt seuraavaan kertaan.

Pylint herjaa muutamasta pikkuasioista, joita voisi korjata. Vastavirta-algoritmi on toteutettu tietoisesti hieman epä-pythonmaisesti. Se on tyyliltään lähempänä "koneoppimisen matematiikka"-kurssilla opittua Matlab-koodista tuttua tyyliä, jossa tulee luotua paljon apumuuttujia. Tällöin prosessin seuraaminen on lähestyttävämpää, mutta pylint ei siitä tykkää.

## Tekoälyn käyttö

Erityisesti projektin alkuvaiheessa jutustelin AI:n kanssa paljonkin tarvittavasta matematiikasta, kun rakentelin ymmärrystä aiheesta Nielsenin kirjan pohjalta. Parhaimmillaan chatGPT onkin mainio sparrauskumppani. Strategianani on käyttää prompteja tyyliin "olenko ymmärtänyt oikein että..", eikä kysyä suoraa vastausta. Tällä tyylillä saattaa itsekin oppia jotain. Koodipuolella tekoälyä on mukava käyttää googlemaisesti tarkistamaan syntax-asioita (esim Numpyyn ja pandas-kirjastoon liittyviä asioita), jos/kun ne eivät olleet aluksi tuttuja. Myös debuggaamisessa ja virheviestien tulkinnassa chatGPT on joskus suurena apuna. Projektin edetessä Ai-jutustelu väheni. En käytä mitään co-pilotteja tms koodieditorissa.

## Huomioita

Projekti oli erittäin opettavainen. Oli mukavaa ensimmäisen opiskeluvuoden päätteeksi päästä käyttämään oikeastaan kaikkea oppimaansa samassa projektissa. Pythonia, lineaarialgebra ja muita oheistyökaluja. Lisäksi tuli opeteltua testaamista, joka oli itselle aika uusi asia, sekä laajempaa dokumentointia, jota ei ole tullut harrastettua ennen tätä. Oli mukavaa tutustua myös Numpyyn, Pandas-kirjastoon ja muihin ML-työkaluihin. Pakko myöntää että ML-kärpänen pääsi vähän puraisemaan ja luulenpa että tästä projektista, tai sen laajentamisesta voisi olla iloa myös kandin työssä ja siitä eteenpäinkin.

## Viitteet

[Michael Nielsenin kirja](http://neuralnetworksanddeeplearning.com/chap1.html)

[3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

[Neuroverkon testaamisesta](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)

[1:O-analyysia](https://lunalux.io/introduction-to-neural-networks/computational-complexity-of-neural-networks/)

[2:wikipedia](https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations)

[3: time complexity of matrix transpose](https://www.ijcsit.com/docs/Volume%207/vol7issue5/ijcsit20160705043.pdf)
