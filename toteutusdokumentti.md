# Toteutusdokumentti

## Ohjelman yleisrakenne
Ohjelma lataa CSV-tiedostosta MNIST-tietokantaan tallennetut 60 000 kuvaa käsin piirretyistä numeroista ja muuntaa 28x28 pikselin kuvat 784 pituisiksi vektoreiksi.
Tämän lisäksi ladataan myös 10 000 testaamiseen tarkoitettua kuvaa. Nämä ovat eri kuvia kuin koulutukseen käytetyt kuvat.
Neuroverkolle annetaan alkuarvoina verkon kerrosten koot. Verkko on yksinkertainen yhden piilokerroksen verkko, joten input-kerros on kokoa 784, piilokerroksen kokoa voi itse hakea, tällä hetkellä sen koko on 60 neuroni ja output-kerros on 10 pituinen vektori, sillä numerot jaotellaan luonnollisesti 10 eri lokeroon. Annetun verkon koon perusteella ohjelma alustaa joka kerrokselle sopivat sattumanvaraiset weight ja bias vektorit, joita lähdetään sitten koulutusvaiheessa päivittämään kohti parempia arvoja. Verkolle voi myös antaa laskennassa käytettäviä aktivaatio- ja error-funktioita. Tämä kyseinen ohjelma käyttää aktivaationa Sigmoidia ja errorina MSE-funktiota(mean square error).

Verkko koulutetaan ajamalla koko tietokanta verkon läpi useita kertoja, näitä kutsutaan eepokeiksi. Eepokkien määrää voi muuttaa koodissa, tällä hetkellä eepokkien määrä on 30. Koulutus tapahtuu 10 samplen "paketeissa" (aka. mini_batch, tämäkin on koodissa muutettavissa). Koulutuksen voisi toteuttaa yksi sample kerrallaan, mutta tällöin jäisi hyödyntämättä Numpy-kirjaston ja Pythonin varsin hyvin optimoituja matriisien kertolaskualgoritmeja. Koulutuksessa samplepaketti muutetaan matriisiksi, joka kerrotaan kerroksittain weight-vektoreilla ja joka sarakkeelle (eli joka samplelle) lisätään kerroksen bias-arvot, lopuksi joka arvo ajetaan aktivaatiofunktion (Sigmoid) läpi, joka normalisoi arvot 0 ja 1 välille. Lopulta aktivoitu kerroksen output-matriisi toimii inputtina seuraavalle kerrokselle ja prosessi toistetaan.

Kun samplet on ajettu verkon läpi, verrataan outputtina saatavia arvoja koulutusdatan mukana tuleviin y-arvoihin, jotka kertovat mitä niiden pitäisi olla. Tästä lasketaan virheen määrä error-funktiolla (mse). Tälle errorille päädytään laskemaan derivoimalla miten verkon eri painot ja biasit vaikuttavat sen määrään. Kun jokaiselle painolle ja biasille on laskettu kulmakerroin (gradient), painoja ja biaseja päivitetään gradientin vastakkaiseen suuntaan pienen nykäyksen verran, nykäyksen kokoa kutsutaan oppimisnopeudeksi (learning rate). Näin verkko oppii.

Miksi piilokerroksen ja samplepakettien kokoa sekä oppimisnopeutta ja samplepakettien määrää voi muuttaa koodissa? Koska näiden arvojen optimointi on olennainen osa verkon kouluttamista. Eri arvoilla saadaan hyvinkin erilaisia tuloksia. Tästä lisää puutteissa ja parannusehdotuksissa.

## Aika-, tila- ja suorituskykyanalyysi
Suorituskyky ei ole neuroverkkojen kannalta kovinkaan olennainen hiottava parametri. Kun verkko on kerran koulutettu, ei sitä tarvitse kouluttaa uudestaan ja käyttö on suht nopeaa. Datan vienti verkon läpi ja luokittelutuloksen saanti vaatii vain muutaman matriisikertolaskun. Koulutusta varten tämä ohjelma on toteutettu hyödyntäen Numpy-kirjaston hyvin optimoituja matriisilaskuja. Koulutuksen voisi toteuttaa myös for-loopilla sample kerrallaan, mutta tällöin matriisilaskun nopeuttavat hyödyt jäävät saamatta.

## Puutteet ja parannusehdotukset
Suurin kysymys koko ohjelmalle on, olisiko pitänyt ratkaista projektiksi määritelty MNIST-jaottelu, vai toteuttaa kokonainen neuroverkkokirjasto. Tämän hetken toteutus on joiltain osin aika lähellä kirjastoa, mutta olennaisilta osin myös täysin räätälöity, yksi ratkaisu MNIST-ongelmaan. Koodista saa aika pienellä vaivalla kirjastomaisemman ja kyvykkään ratkaisemaan muitakin tunnettuja lajitteluongelmia. Suurin rajoite tällä hetkellä on "kovakoodattu" yhden piilokerroksen rajoitus. Muuntamalla vastavirta-algoritmin toteutusta hiukan, saisi piilokerroksen nabla-muuttujat ja painojen ja biasien päivitys toteutettua looppaamalla niin, että kerroksia voisi olla useita. Myös muuntamalla hieman isommin koodin rakennetta voisi joka kerrokselle määritellä omat aktivaatiofunktiot.

Minulla oli aluksi toteutettuna GUI, jossa käyttäjä pystyi piirtämään itse numeroita verkolle tunnistettavaksi, mutta verkko ei ollut kovinkaan hyvä tunnistamaan niitä (ehkä n. 50-60% meni oikein). Tästä kokeilusta jäi kuitenkin olo, että tällainen ominaisuus olisi ihan toteutettavissa, mutta se jäi nyt seuraavaan kertaan.


## Tekoälyn käyttö
Tämä oli ensimmäinen kerta, kun käytin tekoälyä aika paljonkin opiskelun tukena. Parhaimmillaan chatGPT onkin mainio sparrauskumppani. Erityisesti projektin alkuvaiheessa jutustelin AI:n kanssa tarvittavasta matematiikasta, kun rakentelin ymmärrystä aiheesta Nielsenin kirjan pohjalta. Strategianani on käyttää prompteja tyyliin "olenko ymmärtänyt oikein että..", eikä kysyä suoraa vastausta. Tällä tyylillä saattaa itsekin oppia jotain. Koodipuolella tekoälyä on mukava käyttää tarkistamaan syntax-asioita (esim Numpyyn liittyviä), jos/kun ne eivät olleet aluksi tuttuja.

## Huonioita
Projekti oli erittäin opettavainen. Oli mukavaa ensimmäisen opiskeluvuoden päätteeksi päästä käyttämään oikeastaan kaikkea oppimaansa samassa projektissa. Pythonia, lineaarialgebra ja muita oheistyökaluja. Lisäksi tuli opeteltua testaamista, joka oli itselle aika uusi asia, sekä laajempaa dokumentointia, jota ei ole tullut harrastettua juurikaan ennen tätä. Oli mukavaa tutustua myös Numpyyn ja muihin ML-työkaluihin. Pakko myöntää että ML-kärpänen pääsi vähän puraisemaan ja luulenpa että tästä projektista, tai sen laajentamisesta voisi olla iloa myös kandin työssä ja siitä eteenpäinkin.

## Viitteet
[Michael Nielsenin kirja](http://neuralnetworksanddeeplearning.com/chap1.html)
[3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
[Neuroverkon testaamisesta](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)



