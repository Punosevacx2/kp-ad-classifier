# Analiza grešaka i robusnosti

Ovaj dokument analizira **gde i zašto** modeli greše, koliko su otporni na izmenjene naslove, i šta to znači za upotrebu modela. Pored ukupnih metrika, ovo je najkorisniji deo za odluku da li i kako model koristiti u praksi.

## 1. Koliko grešaka uopšte može da se ispravi

Greške baseline-a na ranijoj podeli validacionog skupa (41 greška) ručno su razvrstane u tri grupe:

| grupa | broj | primeri |
|---|---|---|
| **problem u podacima** | 8 | naslov ne pripada grupi u kojoj je označen, ili nijednoj od 10 grupa |
| **dvosmislen naslov bez dovoljno informacija** | 21 | samo ime mesta, brenda ili lika |
| **stvarna greška modela** | 12 | greška u kucanju, retka reč, višeznačnost |

Samo oko **trećine grešaka** je rešivo boljim modelom. Ostatak potiče iz samih podataka.

### Problemi u podacima

| naslov | označen kao | problem |
|---|---|---|
| `#ERROR!` | Apple iPhone | pokvaren podatak iz tabelarnog programa (uklonjen iz konačnog skupa) |
| Stolovi za ugostiteljstvo | Kreveti i dušeci | stolovi nisu ni u jednoj od 10 grupa |
| Autokomanda... renoviran lux stan | Placevi i zemljište | stan, a ne plac |
| Vračar jednoiposoban | Placevi i zemljište | stan |
| Stambeno-poslovni prostor | Placevi i zemljište | poslovni prostor |
| Municija za heftalicu, klamerice | Električne brusilice | spajalice |
| Tepih, nemački proizvođač | Kreveti i dušeci | tepih |
| Marsovac meco | Upaljači, tabakere i pljoske | plišana igračka; model je bio u pravu sa sigurnošću 0,92 |

Iz drugih ispisa: "Fotelja na izvlačenje" i "Dva ista polovna kauča" u grupi Kreveti, "Atestirane plinske boce" u grupi Roštilji, "Aparat za kafu ... Bh" u grupi Apple iPhone, "Samsung Galaxy A55" u grupi Apple iPhone.

### Dvosmisleni naslovi

- imena franšiza bez reči "album" ili "sličice": "Veliki Mapet šou", "Srećni dani", "Flora i fauna", "Fraggle Rock": mogu biti albumi, a mogu biti i plišani likovi,
- samo ime mesta ili poetski naslov: "Lipovica", "Sjajna priča u Budisavi",
- samo brend ili ime proizvoda: "Pierre Cardin", "Glowen Dragon", "Nissan Army 100".

### Stvarne greške modela

| naslov | uzrok |
|---|---|
| **Pljsoka** happy 40th | greška u kucanju ("pljoska") |
| **Eoro** 84 Danska | greška u kucanju ("Euro") |
| PLAVI **BOŠ** originali | fonetski zapis brenda Bosch |
| spring box | engleski naziv (boxspring krevet) |
| **Lezaj** Kika | višeznačnost: ležaj kao krevet ili kao mašinski deo |
| Multifunkcionalni aparat **Bosch** MB-206 | brend koji proizvodi i alate i kućne aparate |
| einhell zagrtač, Sparky FK 65 | redak stručni termin ili brend |

## 2. Zajedničke greške modela

Baseline i mreža su potpuno različiti modeli (linearni nad n-gramima i konvoluciona mreža), a ipak greše uglavnom na istim naslovima:

| poređenje | oba pogrešna | samo baseline pogrešan | samo mreža pogrešna |
|---|---|---|---|
| prva verzija mreže, val | 43 | 13 | 31 |
| konačni ansambl, val | 32 | 24 | 4 |
| konačni ansambl, test | 24 | 11 | 18 |

To nezavisno potvrđuje da je značajan deo grešaka posledica podataka: pogrešnih labela, dvosmislenih naslova i predmeta van 10 grupa. Na test skupu 24 greške pravi oba modela, što je približno "pod" ispod koga se ne može ići bez čišćenja labela.

## 3. Obrasci po grupama

**Plišane igračke kao "podrazumevana" grupa.** Kod svih modela ta grupa ima najnižu preciznost. Kada naslov nema jasan signal (ime lika, brend, ime mesta), modeli ga češće svrstavaju u Plišane igračke, jer su to često kratki naslovi sa imenima ("Pufi", "Anđelčić", "Grof Lisa"). Najčešća zamena kod svih modela je Sličice i albumi → Plišane igračke: obe grupe dele vokabular likova i franšiza (Disney, Pokemon, Mapet).

**Grupe sa jasnim vokabularom** (Samsung futrole, Apple iPhone, Placevi) imaju F1 iznad 0,995. Hipoteza iz analize podataka da će se Apple iPhone i Samsung futrole mešati **nije se potvrdila**: ovaj par se ne pojavljuje među najčešćim zamenama ni kod jednog modela.

**Sličnost vokabulara iz analize podataka predvidela je strukturu grešaka.** Parovi grupa sa najvećom Jaccard sličnošću vokabulara (Sličice ↔ Upaljači 0,21, Plišane ↔ Sličice 0,21) upravo su parovi koje modeli najčešće mešaju.

## 4. Kako razmišljaju modeli: šta mreža rešava, a šta kvari

Poređenje prve verzije mreže sa baseline-om na validacionom skupu (pre label smoothing-a):

**Mreža tačna, baseline pogrešan (13 naslova):**
- "Dva **samca**" → Kreveti: grana karaktera povezuje "samca" sa "krevet samac", iako je u vokabularu reči to drugi token,
- "SAHARA PLUS **boxspring**", "Grof Lisa" → Kreveti,
- u 8 od 13 slučajeva baseline je birao Plišane igračke, pa mreža manje pada na "podrazumevanu" grupu.

**Baseline tačan, mreža pogrešna (31 naslov), često sa velikom sigurnošću:**

| naslov | mreža predviđa | sigurnost |
|---|---|---|
| Silikonska futrola iqos 3 duo | Samsung | 1,000 |
| Haba zvečka i zemljina kugla kuća put auto pile drvo | Placevi | 0,992 |
| Sunčica kids hotels došla u Moskvu | Kreveti | 0,987 |
| violet klokan liga Evrope | Sličice | 0,977 |

Obrazac: **jedna jaka fraza odlučuje o svemu**. "Silikonska futrola" je skoro uvek Samsung, pa mreža ignoriše `iqos`; "kuća, put, drvo" zvuče kao plac, pa mreža ignoriše "zvečka". Max pooling prenosi samo najjaču aktivaciju svakog filtera, dok logistička regresija sabira doprinose svih reči. Ova analiza je direktno dovela do uvođenja max + mean pooling-a i label smoothing-a.

**Sigurna greška i u konačnom modelu.** "Pufi sa kapom" (plišana igračka) konačni ansambl svrstava u Električne brusilice sa sigurnošću 0,983. Pufi je redak naziv koji nijedna grana ne prepoznaje, pa model koristi jedini prepoznatljiv deo naslova. Kalibracija važi u proseku, ali pojedinačne greške i dalje mogu biti sigurne.

## 5. Prečice (shortcut learning)

Tokeni iz normalizacije proveravani su preko težina baseline modela:

| token | najveća težina | tumačenje |
|---|---|---|
| `<phone>` | 0,86 (Apple iPhone) | umeren signal; zamena konkretnih brojeva tokenom sprečila je da model nauči broj otkupljivača |
| `<id>` | 3,41 (Placevi i zemljište) | jak signal; šifre oglasa koriste agencije za nekretnine |

`<id>` je svestan kompromis: to je realan signal, ali se odnosi na to **ko** je postavio oglas, a ne **šta** se prodaje. Oglas iz druge grupe sa "šifra: 12345" bio bi gurnut ka Placevima.

Primer naučene prečice u podacima: "Aparat za kafu ... **Bh**" je označen kao Apple iPhone (pogrešna labela), a mreža ga je "pogodila" sa 0,90, verovatno zato što je `bh` (battery health) čest u iPhone oglasima.

## 6. Testovi robusnosti

Implementirano u `src/kp_classifier/robustness.py`. Od validacionog i test skupa prave se izmenjene verzije naslova, a pad tačnosti se meri na istim naslovima:

| varijanta | izmena |
|---|---|
| `typo_1` | jedna reč (od bar 4 slova) dobija grešku u kucanju: zamena, brisanje, dupliranje slova ili zamena mesta susednih slova |
| `typo_2` | isto, u dve reči |
| `shuffled_words` | reči nasumično ispremeštane |
| `drop_keyword` | izbačena reč koja najjače ukazuje na tačnu grupu, određena iz statistike trening skupa, nezavisno od modela |
| `has_unknown_word` | originalni naslovi sa bar jednom reči van vokabulara |
| `no_unknown_word` | ostali originalni naslovi |

Primer: "kreveti od punog drveta" → `typo_2`: "kreveti od pnog dreta", `drop_keyword`: "od punog drveta".

Poređeni modeli: baseline, konačni ansambl, jedna mreža iz ansambla (`maxmean_ls01_s42`) i **ista mreža bez grane karaktera** (`words_mm_ls_s42`).

### Broj grešaka, test skup

| varijanta | n | baseline | ansambl | mreža (reči + karakteri) | mreža (samo reči) |
|---|---|---|---|---|---|
| original | 4.150 | **35** | 42 | 43 | 56 |
| typo_1 | 4.142 | **81** | 91 | 105 | 187 |
| typo_2 | 4.142 | **139** | 170 | 197 | 507 |
| shuffled_words | 4.124 | **33** | 39 | 43 | 52 |
| drop_keyword | 4.122 | 427 | 348 | 380 | **326** |
| has_unknown_word | 1.476 | **27** | 33 | 33 | 48 |
| no_unknown_word | 2.674 | **8** | 9 | 10 | **8** |

### Broj grešaka, validacioni skup

| varijanta | n | baseline | ansambl | mreža (reči + karakteri) | mreža (samo reči) |
|---|---|---|---|---|---|
| original | 4.150 | 56 | **36** | 39 | 51 |
| typo_1 | 4.139 | **99** | 100 | 103 | 177 |
| typo_2 | 4.139 | **168** | 196 | 224 | 497 |
| shuffled_words | 4.127 | 52 | 38 | **36** | 49 |
| drop_keyword | 4.121 | 457 | 375 | 383 | **318** |
| has_unknown_word | 1.480 | 45 | **29** | **29** | 45 |
| no_unknown_word | 2.670 | 11 | 7 | 10 | **6** |

### Pad tačnosti u odnosu na iste originalne naslove, test skup (procentni poeni)

| varijanta | baseline | ansambl | mreža (reči + karakteri) | mreža (samo reči) |
|---|---|---|---|---|
| typo_1 | 1,11 | 1,18 | 1,50 | 3,19 |
| typo_2 | 2,51 | 3,09 | 3,72 | **10,91** |
| drop_keyword | 9,53 | 7,52 | 8,25 | 6,70 |
| shuffled_words | −0,02 | 0,02 | 0,07 | 0,02 |

### Zaključci

**1. Grana karaktera je ključna za otpornost na greške u kucanju.** Uz dve pogrešno otkucane reči, mreža samo sa rečima pada na 87,8% tačnosti, a ista mreža sa granom karaktera ostaje na 95,2%, sa 61% manje grešaka. Kada "brusilica" postane "brusilcia", grana reči vidi `<unk>`, a grana karaktera i dalje prepoznaje "brus" i "sil". Isti rezultat dobijen je na oba skupa.

**2. Baseline je najotporniji na veštačke greške u kucanju.** Linearni model nad desetinama hiljada karakterskih n-grama **sabira** njihove doprinose, pa jedno pogrešno slovo pokvari samo nekoliko n-grama. Isti rezultat na oba skupa.

**3. Mreže bolje koriste kontekst kada nedostaje ključna reč.** Bez reči poput "krevet" ili "plac", ansambl pravi 348 grešaka, a baseline 427. Isti rezultat na oba skupa.

**4. Mreža samo sa rečima je najbolja bez ključne reči.** Word dropout tokom treniranja briše reči, pa mreža samo sa rečima mora da nauči da koristi ceo naslov. U hibridu grana karaktera uvek vidi ceo tekst, pa nastavlja da se oslanja na ključnu reč preko karaktera. Predlog za dalji rad: brisati istu reč i iz niza karaktera tokom treniranja.

**5. Redosled reči ne utiče na rezultat** ni kod jednog modela (manje od 0,2 pp). Naslovi se ponašaju kao skup ključnih reči, a ne kao rečenica. Zato je model nad n-gramima toliko jak, i zato sekvencijalni modeli (LSTM, Transformer) verovatno ne bi doneli korist.

**6. Nepoznate reči: rezultat se nije ponovio.** Na validacionom skupu mreža je bila znatno bolja od baseline-a na naslovima sa nepoznatim rečima (29 naspram 45 grešaka), a na test skupu nije (33 naspram 27). Ovaj nalaz ne smatramo pouzdanim. Pouzdano je samo da je grana karaktera bolja od grane reči na takvim naslovima (33 naspram 48 na testu).

## 7. Preporuke za upotrebu

1. **Prag sigurnosti.** Uz prag 0,99 model automatski klasifikuje oko 96% oglasa sa tačnošću od 99,9%. Ostatak treba poslati na proveru ili korisniku ponuditi izbor.
2. **Predlog dve grupe.** Tačna grupa je među dve najverovatnije u 99,5% slučajeva, pa je predlog dve grupe pri postavljanju oglasa praktičan način upotrebe.
3. **Klasa "ostalo" ili detekcija nepoznatog.** Model uvek bira jednu od 10 grupa. Na pravoj platformi, sa stotinama grupa, potrebna je opcija "nijedna od ponuđenih": dodatna klasa sa primerima iz drugih grupa, ili posebna detekcija naslova van distribucije. Prag sigurnosti to rešava samo delimično.
4. **Čišćenje labela.** Greške u kojima je model najsigurniji su najčešće pogrešne labele (`*_errors.csv` je sortiran po sigurnosti). Pregled tih primera je najjeftiniji način da se poboljšaju i podaci i procena tačnosti.
5. **Praćenje u upotrebi.** Udeo naslova ispod praga i udeo nepoznatih reči su jednostavni signali da se jezik oglasa promenio i da model treba ponovo obučiti.

## 8. Reprodukcija

```bash
python -m kp_classifier.robustness --models baseline_tfidf words_mm_ls_s42 maxmean_ls01_s42 ens_mm_ls_cal
python -m kp_classifier.robustness --models baseline_tfidf words_mm_ls_s42 maxmean_ls01_s42 ens_mm_ls_cal \
  --split test --out-dir reports/robustness_test
```

Rezultati: `reports/robustness/` i `reports/robustness_test/` (tabela, grafik i primeri izmenjenih naslova). Liste grešaka svakog modela, sortirane po sigurnosti: `reports/runs/<model>/<skup>_errors.csv`.
