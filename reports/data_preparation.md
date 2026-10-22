# Priprema trening podataka

Ovaj dokument opisuje put od tabele `ml_dataset` u bazi do skupova za treniranje, validaciju i testiranje, kao i razloge za svaku odluku. Detaljna eksplorativna analiza nalazi se u [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb).

## 1. Izvor podataka

Tabela `ml_dataset` izvezena je skriptom `src/kp_classifier/data/export_mysql.py` u `data/raw/ml_dataset.parquet`. Ostale tabele iz baze nisu korišćene.

| kolona | uloga |
|---|---|
| `name` | naziv oglasa, **jedini ulaz u model** |
| `group_id`, `group_name` | ciljna klasa |
| `category_id`, `category_name` | nadređena kategorija grupe |

Skup ima 50.000 naslova u 10 grupa po 5.000, bez nedostajućih vrednosti. Mapiranje grupa ↔ kategorija je 1:1, pa kategorija ne nosi informaciju dodatnu u odnosu na grupu i nije korišćena ni kao ulaz ni kao pomoćni cilj. Kao ulaz bi u svakom slučaju bila neprihvatljiva, jer model u upotrebi dobija samo naziv oglasa.

## 2. Nalazi iz analize koji određuju pripremu

| nalaz | posledica |
|---|---|
| 16% naslova su duplikati posle osnovne normalizacije, neravnomerno po grupama (1,2% – 39,8%) | deduplikacija pre podele, inače curenje podataka iz trening u test skup |
| ista reč se javlja sa i bez dijakritika (`dusek` 568, `dušek` 535) | svođenje na ASCII |
| ćirilica u 0,1% naslova | preslovljavanje u latinicu |
| 7,2% naslova grupe Apple iPhone sadrži broj telefona, često isti broj otkupljivača | zamena tokenom `<phone>`, da model ne bi naučio broj umesto sadržaja |
| šifre oglasa ("Šifra: 528492") gotovo isključivo u grupi Placevi i zemljište | zamena tokenom `<id>` |
| naslovi su odsečeni na 60 karaktera, najduži ima 19 reči | maksimalna dužina ulaza 60 karaktera i 20 reči, bez odsecanja |
| 47% jedinstvenih tokena javlja se samo jednom | vokabular sa pragom frekvencije i grana karaktera u modelu |
| 76% čestih tokena gotovo jednoznačno određuje grupu | stop-reči se ne uklanjaju |
| česti brojevi su jak signal (`506` → IMT motokultivatori, `125` → brusilice) | brojevi se ne uklanjaju, nego se retki svode na token oblika |

## 3. Normalizacija teksta

Normalizacija je implementirana u `src/kp_classifier/data/preprocess.py`, u funkciji `normalize_text`. **Ista funkcija se koristi pri treniranju i pri predikciji**, pa model u upotrebi dobija tekst u istom obliku kao tokom učenja.

Koraci, redom:

| korak | primer |
|---|---|
| 1. Unicode NFKC normalizacija | ujednačava različite zapise istog znaka |
| 2. ćirilica → latinica | `Француски лежај` → `Francuski ležaj` |
| 3. ćirilično `х` između cifara → `x` | `113х225` → `113x225` (inače bi postalo `113h225`) |
| 4. broj telefona → `<phone>` | `OTKUP iPhone 065/280-20-20` → `OTKUP iPhone <phone>` |
| 5. šifra oglasa → `<id>` | `Plac 2101m2! Sifra: 528492` → `Plac 2101m2! <id>` |
| 6. mala slova | `NA STANJU` → `na stanju` |
| 7. uklanjanje dijakritika | `dušek bračni` → `dusek bracni`, `đ` → `dj` |
| 8. uklanjanje interpunkcije i simbola | `(Rezervisan)iPhone 6s!` → `rezervisan iphone 6s` |
| 9. sažimanje razmaka | |

Redosled je bitan: telefon se zamenjuje **pre** uklanjanja interpunkcije, jer su `/`, `-` i razmaci deo zapisa broja. Kada bi interpunkcija nestala prva, `065/280-20-20` bi postao niz od četiri nezavisna broja.

### Prepoznavanje telefona i iterativna provera

Pravila su razvijana iterativno: pravilo → primena na svih 50.000 naslova → ručni pregled slučajeva u kojima se novi i jednostavni detektor ne slažu → ispravka → test u `tests/test_preprocess.py`. Svaki pronađeni problematičan format postao je test, pa se ista greška ne može vratiti.

Nalazi tokom provere:

- **Propušteni telefoni:** zapisi sa kosom crtom neposredno ispred broja (`Starlight/064/954-23-97`) i sa tačkama ili crticama kao separatorima (`066.366.669`, `061-5-666-555`).
- **Kataloški brojevi koji liče na telefone:** Bosch obeležava alate brojevima `0 601 xxx xxx` i `0 603 xxx xxx`, koji su formatom identični srpskim mobilnim brojevima sa pozivnim 060/061 (18 lažnih pozitiva u grupi Električne brusilice). Isto važi za ETA (`0168 90000`, 5 naslova u grupi Roštilji). Sam broj nije dovoljan za razlikovanje, pa se koristi kontekst: broj se ne menja ako naslov sadrži naziv brenda ili serije (`bosch`, `gws`, `gss`..., `eta`) i broj odgovara formatu kataloškog broja tog brenda (lista `BRAND_ARTICLE_RULES`).
- **Lažni `<id>`:** model alata `TC-ID 750 E` (Einhell) prepoznavan je kao šifra oglasa. Pošto se u podacima šifra oglasa javlja isključivo uz reč "šifra", pravilo je suženo na nju.

Konačan rezultat: **382 telefona** (364 u grupi Apple iPhone) i **213 šifri oglasa**, sve u grupi Placevi i zemljište. Nijedan naslov ne postaje prazan posle normalizacije.

## 4. Čišćenje

Implementirano u `src/kp_classifier/data/split.py`, funkcija `clean`.

| korak | uklonjeno redova |
|---|---|
| polazni skup | 50.000 |
| pokvareni redovi (vrednost `#ERROR!` iz tabelarnog programa) | 1 |
| prazni posle normalizacije | 0 |
| konflikti: isti normalizovan naslov u više grupa (9 naslova) | 26 |
| duplikati posle pune normalizacije | 8.471 |
| **preostalo** | **41.502** |

Važne odluke:

- **Deduplikacija se radi posle pune normalizacije.** Normalizacija spaja naslove koji su se razlikovali samo po dijakriticima, pismu ili velikim slovima ("Dušek" = "dusek" = "Душек"). Posle osnovne normalizacije duplikata je 7.954, a posle pune 8.479 (u tabeli je 8.471 jer se konflikti uklanjaju pre deduplikacije).
- **Konflikti se uklanjaju pre deduplikacije.** U suprotnom bi deduplikacija zadržala jednu kopiju sa proizvoljnom labelom. Svi konflikti su generički naslovi bez informacije ("prodajem", "kuca", "na prodaju").
- **Razmotrena alternativa:** zadržati duplikate i deliti skup po grupama identičnih naslova (`GroupShuffleSplit`), čime se čuva realna frekvencija naslova. Izabrana je deduplikacija, jer daje pošteniju procenu na neviđenim naslovima.
- **Determinizam:** pre deduplikacije redovi se sortiraju po grupi i tekstu, pa rezultat ne zavisi od redosleda u kome baza vraća redove.

Raspodela klasa posle čišćenja:

| grupa | pre | posle | uklonjeno |
|---|---|---|---|
| Samsung, futrole, maske i folije | 5.000 | 4.935 | 1,3% |
| Električne brusilice | 5.000 | 4.815 | 3,7% |
| Apple iPhone | 5.000 | 4.622 | 7,6% |
| Placevi i zemljište | 5.000 | 4.581 | 8,4% |
| Motokultivatori, delovi i oprema | 5.000 | 4.541 | 9,2% |
| Sličice i albumi | 5.000 | 4.097 | 18,1% |
| Plišane igračke | 5.000 | 3.865 | 22,7% |
| Upaljači, tabakere i pljoske | 5.000 | 3.627 | 27,5% |
| Kreveti i dušeci | 5.000 | 3.465 | 30,7% |
| Roštilji, grilovi i friteze | 5.000 | 2.954 | 40,9% |

Duplikati su najčešći u grupama sa kratkim generičkim naslovima ("krevet" se ponavlja 298 puta, "friteza", "upaljac"), a najređi tamo gde naslov skoro uvek sadrži model ili šifru. Odnos najveće i najmanje klase je 1,67, što je blaga neravnoteža. Zbog nje se pored tačnosti prati i macro F1.

## 5. Podela na skupove

Svaka grupa se deli zasebno, u odnosu 80 / 10 / 10:

| skup | broj naslova |
|---|---|
| trening | 33.202 |
| validacioni | 4.150 |
| test | 4.150 |

- **Stratifikacija:** validacioni i test skup imaju identičan broj primera svake grupe, a najveće odstupanje udela grupe između tri skupa je 0,016 procentnih poena.
- **Seed po grupi:** svaka grupa ima sopstveni generator slučajnih brojeva (`[seed, label]`), pa izmena u jednoj grupi (npr. uklanjanje jednog reda) ne menja podelu ostalih grupa.
- **Provera curenja:** posle podele se proverava da se nijedan normalizovan naslov ne nalazi u dva skupa, inače skript prekida rad.
- **Mapiranje labela** (`label_map.json`, indeksi po rastućem `group_id`):

| indeks | group_id | grupa |
|---|---|---|
| 0 | 232 | Placevi i zemljište |
| 1 | 405 | Kreveti i dušeci |
| 2 | 489 | Apple iPhone |
| 3 | 589 | Sličice i albumi |
| 4 | 769 | Upaljači, tabakere i pljoske |
| 5 | 973 | Električne brusilice |
| 6 | 1142 | Plišane igračke |
| 7 | 1277 | Motokultivatori, delovi i oprema |
| 8 | 2173 | Samsung, futrole, maske i folije |
| 9 | 2663 | Roštilji, grilovi i friteze |

Uloge skupova:
- **trening:** učenje težina modela i vokabulara,
- **validacioni:** svi izbori (hiperparametri, arhitektura, izbor epohe, temperatura kalibracije),
- **test:** jedna konačna ocena izabranih modela, posle zamrzavanja modela.

## 6. Tokenizacija

Implementirano u `src/kp_classifier/data/tokenizer.py`. Vokabulari se grade **isključivo na trening skupu**.

**Vokabular reči: 9.482 tokena**

~~~
0      <pad>                         dopuna do 20 tokena
1      <unk>                         reč koje nema u vokabularu
2–5    <num1> <num2> <num3> <num4+>  retki brojevi, po broju cifara
6–7    <phone> <id>                  tokeni iz normalizacije
8+     reči sa bar 2 pojavljivanja u trening skupu, od najčešće ka ređoj
~~~

Za svaku reč u naslovu: ako je u vokabularu, dobija svoj indeks (uključujući česte brojeve kao `506` ili `125`); ako je čist broj van vokabulara, dobija token oblika (`7` → `<num1>`, `4581` → `<num4+>`); sve ostalo postaje `<unk>`.

**Vokabular karaktera: 41 znak** (4 rezervisana, razmak, 26 slova, 10 cifara). Tokeni `<phone>` i `<id>` u grani karaktera predstavljeni su jednim posebnim znakom, da ne bi trošili mesto i ličili na reč "phone".

**Dužine:** 20 reči i 60 karaktera. Odseca se 0,05% naslova u validacionom skupu (preslovljavanje `đ` → `dj` produžava tekst).

**Pokrivenost:**

| | trening | validacioni |
|---|---|---|
| udeo `<unk>` tokena | 5,2% | 8,1% |
| naslova sa bar jednim `<unk>` | 24,2% | **35,7%** |
| udeo tokena `<numN>` | 0,5% | 0,9% |

Više od trećine novih naslova sadrži reč koju grana reči ne prepoznaje. To je glavni kvantitativni argument za granu karaktera (vidi [arhitekturu](architecture.md)).

Prag frekvencije 2 je nameran: reči viđene samo jednom ne ulaze u vokabular, pa se `<unk>` javlja i u trening skupu (24% naslova) i model uči šta da radi sa nepoznatom rečju. Da je vokabular sadržao sve trening reči, `<unk>` se u treniranju ne bi pojavio, a u validaciji bi bio u 36% naslova.

## 7. Šta namerno nije urađeno

| odluka | obrazloženje |
|---|---|
| stop-reči se ne uklanjaju | standardna lista bi uklonila `u`, `na` i `za`, a `u` je 64% vezan za grupu Placevi ("plac u Obrenovcu"); samo 0,5% čestih tokena je stvarno ravnomerno raspoređeno po grupama |
| brojevi se ne uklanjaju | česti brojevi su među najjačim signalima; nijedan naslov nije sastavljen samo od brojeva |
| kategorija se ne koristi | 1:1 sa grupom, a nije dostupna u upotrebi |
| bez ručnog ispravljanja labela | pogrešne labele postoje (npr. "Samsung Galaxy A55" u grupi Apple iPhone), ali se ne ispravljaju ručno da evaluacija ne bi bila pristrasna; analizirane su u [analizi grešaka](error_analysis.md) |

## 8. Napomena o veličini skupova

Isti baseline model postigao je 98,65% tačnosti na jednoj nasumičnoj podeli i 99,01% na drugoj. Sa 4.150 primera, 95% interval poverenja za tačnost od ~99% je širok oko ±0,3 procentna poena, što je reda veličine razlike između modela. Zato su u evaluaciji uz svaku metriku prijavljeni intervali poverenja, a modeli su poređeni McNemar testom na istim primerima (vidi [metrike](metrics.md)).

## 9. Reprodukcija

~~~bash
python -m kp_classifier.data.export_mysql
python -m kp_classifier.data.split
python -m kp_classifier.data.tokenizer
~~~

Rezultat: `data/processed/{train,val,test}.parquet`, `label_map.json`, `split_stats.json` i `vocab.json`.
