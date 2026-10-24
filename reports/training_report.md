# Izveštaj sa treniranja

Ovaj dokument opisuje kako je model obučavan, koje su odluke donete na putu do konačnog modela i zašto, i kakav je rezultat na test skupu. Eksperimenti su prikazani hronološki, jer je svaki korak proizašao iz analize prethodnog.

Arhitektura je detaljno opisana u [architecture.md](architecture.md), a metrike u [metrics.md](metrics.md).

## 1. Postavka

| stavka | vrednost |
|---|---|
| podaci | 41.502 naslova posle čišćenja ([priprema podataka](data_preparation.md)) |
| podela | trening 33.202, validacioni 4.150, test 4.150 |
| hardver | Apple M1, PyTorch 2.14 sa MPS backendom |
| trajanje | ~5 s po epohi, 4–5 minuta po mreži |
| glavna metrika | macro F1, uz tačnost, log loss, ECE i krivu rizik–pokrivenost |
| poređenje modela | McNemar test na istim primerima, bootstrap intervali poverenja |

**Protokol:** sve odluke (hiperparametri, arhitektura, izbor epohe, temperatura kalibracije) donete su na validacionom skupu. Test skup je korišćen **jednom**, posle zamrzavanja konačnog modela. Ukupno je izvedeno oko 20 runova.

## 2. Faza 1: baseline

Pre neuronske mreže postavljena je referentna tačka: **TF-IDF + logistička regresija**, nad rečima i parovima reči i nad karakterskim n-gramima dužine 2–5 (87.036 osobina). Jedini hiperparametar je jačina regularizacije `C`.

| C | val macro F1 | val accuracy | val log loss |
|---|---|---|---|
| 1 | 0,9861 | 98,67% | 0,1003 |
| 3 | 0,9859 | 98,65% | 0,0675 |
| 10 | 0,9862 | 98,67% | 0,0510 |
| 30 | 0,9859 | 98,65% | 0,0451 |
| **100** | 0,9860 | 98,65% | **0,0450** |
| 300 | 0,9851 | 98,58% | 0,0510 |

**Pravilo izbora:** među vrednostima čiji je macro F1 najviše 0,0005 ispod najboljeg (oko 2 primera), bira se ona sa najmanjim log loss-om. Macro F1 je ravan za `C` od 1 do 100, jer se razlike svode na jedan do dva primera, dok log loss ima jasan minimum na 100 i ponovo raste na 300. Pravilo je definisano pre nego što su rezultati pogledani, i nije menjano posle.

Rezultat: **98,65% tačnosti, 56 grešaka, ECE 0,0024** na validacionom skupu. Trivijalni model (uvek najveća klasa) ima 11,9%.

Usput dva važna zapažanja:
- **Baseline je vrlo jak.** Naslovi su kratki, a grupe se razlikuju po ključnim rečima, pa je problem skoro linearno separabilan.
- **Šum zbog uzorka je velik.** Isti baseline je na ranijoj nasumičnoj podeli imao 41 grešku (99,01%), a na konačnoj 56 (98,65%). Razlika od 0,36 pp odgovara širini 95% intervala poverenja (±0,33 pp). Odatle odluka da se modeli porede samo uz statističke testove.

## 3. Faza 2: prva verzija mreže

Hibridni CNN sa granom reči i granom karaktera, samo max pooling, bez label smoothing-a (`configs/textcnn.yaml`). Treniranje: AdamW (lr 0,002, weight decay 0,0001), batch 128, smanjenje learning rate-a na plato, rano zaustavljanje po val loss-u posle 5 epoha bez poboljšanja.

**Rezultat: 74 greške (98,22%), lošije od baseline-a.** McNemar: baseline tačan na 31 naslovu na kome mreža greši, mreža na 13 na kojima baseline greši, p = 0,0096. Baseline je značajno bolji.

**Dijagnoza iz krivih učenja:** val loss je imao minimum u epohi 11 (0,0671), a val tačnost je nastavila da raste do 98,43% u epohi 16. Kada loss raste dok tačnost raste, model postaje **previše siguran u pogrešne odgovore**: nekoliko grešaka sa verovatnoćom 0,999 nosi veliki loss. Posledica je i da rano zaustavljanje, vođeno val loss-om, prekida treniranje prerano.

**Dijagnoza iz analize grešaka** (poređenje grešaka mreže i baseline-a):
- mreža greši sa sigurnošću iznad 0,95 kada **jedna jaka fraza nadjača ostatak naslova** ("Silikonska futrola iqos 3 duo" → Samsung, 1,000), dok baseline sabira doprinose svih reči,
- 43 od 56 grešaka baseline-a pravi i mreža, pa je većina grešaka u samim podacima.

Odatle dve hipoteze: (1) kazniti preteranu sigurnost **label smoothing-om**, (2) uz max pooling dodati **mean pooling**, koji prenosi ukupnu "masu dokaza" iz naslova.

## 4. Faza 3: eksperimenti (jedan run po konfiguraciji)

Sve na validacionom skupu, najviše 30 epoha:

| run | promena u odnosu na prvu verziju | greške |
|---|---|---|
| hybrid_cnn | prva verzija, seed 42 | 74 |
| hybrid_seed1 | seed 1 | 72 |
| hybrid_seed2 | seed 2 | **88** |
| cnn_words | samo grana reči | 73 |
| cnn_chars | samo grana karaktera | 95 |
| hybrid_maxmean | max + mean pooling | 75 |
| hybrid_ls01 | label smoothing 0,1 | 49 |
| hybrid_maxmean_ls01 | max + mean pooling + label smoothing | **39** |
| *baseline* | | *56* |

Zaključci:
- **Seed sam menja rezultat za čak 16 grešaka** (72–88). Razlike manje od ~10 grešaka između pojedinačnih runova su u šumu, pa svaki zaključak mora biti proveren na više seed-ova.
- **Label smoothing** ima efekat daleko veći od šuma (74 → 49).
- Bez label smoothing-a ni max + mean pooling ni grana karaktera ne donose merljivu razliku (75 i 73 naspram 74).
- Sama grana karaktera nije dovoljna (95).

McNemar protiv baseline-a: `hybrid_maxmean_ls01` 21 naspram 4, p = 0,0009 (Bonferroni korekcija za 9 konfiguracija: p ≈ 0,008); `hybrid_ls01` 17 naspram 10, p = 0,25.

Oba runa sa label smoothing-om stigla su do limita od 30 epoha dok je val loss još opadao. Label smoothing je promenio dinamiku treniranja: val loss sada opada zajedno sa tačnošću, pa rano zaustavljanje više ne prekida treniranje prerano.

## 5. Faza 4: provera na tri seed-a

Dve najbolje konfiguracije, sa limitom podignutim na 60 epoha, svaka sa seed-ovima 42, 1 i 2:

| konfiguracija | seed 42 | seed 1 | seed 2 | prosek | najbolje epohe |
|---|---|---|---|---|---|
| bez label smoothing-a (faza 3) | 74 | 72 | 88 | 78,0 | 11, 12, 8 |
| label smoothing | 45 | 50 | 51 | 48,7 | 57, 51, 48 |
| **max + mean + label smoothing** | **39** | **45** | **38** | **40,7** | 53, 44, 47 |

- **Label smoothing** smanjuje prosečan broj grešaka sa 78 na 48,7, a rasipanje između seed-ova sa 16 na 6.
- **Max + mean pooling** je, uz label smoothing, bolji na **sva tri seed-a** (39 < 45, 45 < 50, 38 < 51). Bez label smoothing-a nije pomogao. Verovatno objašnjenje je da previše samouveren model ignoriše prosečne osobine, jer mu je dovoljan jedan jak signal iz max pooling-a.
- Duže treniranje nije bilo presudno: seed 42 ima 39 grešaka i sa 30 i sa 60 epoha.

Kontrolni run: ista konačna konfiguracija **bez grane karaktera** (`words_mm_ls_s42`) ima **51 grešku** naspram 39 sa granom karaktera.

## 6. Faza 5: ansambl i kalibracija

**Ansambl:** prosek verovatnoća tri mreže sa seed-ovima 42, 1 i 2.

| | greške (val) |
|---|---|
| članovi | 39, 45, 38 |
| **ansambl** | **36** |

**Kalibracija:** mreže sa label smoothing-om su nedovoljno samouverene (ECE 0,104). Temperature scaling sa `T` izabranim na validacionom skupu (minimum log loss-a) dao je **T = 0,508**:

| | bez kalibracije | sa kalibracijom |
|---|---|---|
| greške | 36 | 36 |
| log loss | 0,140 | **0,039** |
| ECE | 0,104 | **0,003** |

Predikcije se ne menjaju, menja se samo sigurnost.

**Odbačena varijanta:** ansambl tri mreže i baseline-a (39 grešaka, lošije od samih mreža). Baseline daje znatno oštrije verovatnoće od mreža sa label smoothing-om, pa pri prostom proseku dominira, i to upravo na primerima na kojima sam greši.

**Konačni model:** `ens_mm_ls_cal`. Na validacionom skupu: 99,13% tačnosti, 36 grešaka, McNemar protiv baseline-a 24 naspram 4, p = 0,0002.

## 7. Rezultat na test skupu

| | baseline | **ansambl (konačni model)** |
|---|---|---|
| accuracy | 99,16% [98,89 – 99,42] | 98,99% [98,67 – 99,28] |
| macro F1 | 0,9912 | 0,9892 |
| greške | 35 | 42 |
| log loss | 0,0382 | 0,0382 |
| top-2 accuracy | 99,49% | 99,52% |
| ECE | 0,0058 | **0,0033** |
| pokrivenost uz 99,9% tačnosti | 95,9% | 95,8% |

McNemar: oba tačna 4.097, oba pogrešna 24, samo baseline tačan 18, samo ansambl tačan 11, **p = 0,26**.

**Na test skupu se modeli statistički ne razlikuju.** Prednost mreže izmerena na validacionom skupu se nije ponovila.

### Zašto se prednost nije ponovila

1. **Validacioni skup je "potrošen" biranjem.** Mreža je prošla kroz oko 20 runova i nekoliko izbora arhitekture i regularizacije, svi ocenjeni na istih 4.150 naslova. Izabrana konfiguracija je delom zaista bolja, a delom prilagođena baš tom skupu. Bonferroni korekcija štiti od slučajnosti pojedinačnog poređenja, ali ne i od toga da su sve odluke donete gledajući isti skup. Baseline je imao samo jedan hiperparametar i bio je mnogo manje izložen ovom efektu.
2. **Validacioni skup je slučajno bio teži.** Isti baseline ima 56 grešaka na validacionom i 35 na test skupu. Oba skupa su nasumični uzorci iz istih podataka.

Upravo zato postoji test skup koji nijedna odluka nije videla. Model posle ovoga **nije menjan**, jer bi tada isti problem nastao i sa test skupom.

### Šta se pouzdano ponovilo na testu

| nalaz | validacioni | test |
|---|---|---|
| grana karaktera smanjuje greške (samo reči → reči + karakteri) | 51 → 39 | 56 → 43 |
| isto, uz greške u kucanju u dve reči | 497 → 224 | 507 → 197 |
| mreža bolje koristi kontekst kada nedostaje ključna reč (baseline → ansambl) | 457 → 375 | 427 → 348 |
| baseline je otporniji na greške u kucanju (dve reči) | 168 naspram 196 | 139 naspram 170 |
| redosled reči ne utiče na rezultat | < 0,2 pp | < 0,1 pp |
| kalibracija se prenosi na nove podatke (ECE) | 0,003 | 0,003 |

### Šta se nije ponovilo

- ukupna prednost mreže nad baseline-om (val p = 0,0002, test p = 0,26),
- prednost mreže na naslovima sa nepoznatim rečima (val 29 naspram 45 grešaka, test 33 naspram 27).

Detalji testova robusnosti su u [analizi grešaka](error_analysis.md).

## 8. Zaključak

Projektovana neuronska mreža, obučena od nule, klasifikuje naslove oglasa sa **98,99% tačnosti** na neviđenim podacima, dobro je kalibrisana (ECE 0,003) i uz prag sigurnosti 0,99 može automatski da klasifikuje oko 96% oglasa sa tačnošću od 99,9%.

Na kratkim naslovima, čija je grupa uglavnom određena ključnim rečima, mreža dostiže nivo jakog linearnog modela nad n-gramima, ali ga ne nadmašuje na statistički značajan način. Svaka arhitektonska odluka (grana karaktera, max + mean pooling, label smoothing, ansambl, kalibracija) proverena je eksperimentom, a vrednost grane karaktera potvrđena je i na test skupu.

## 9. Šta bi bilo urađeno drugačije

- **Unakrsna validacija za izbor modela.** Sa 4.150 primera po skupu, šum je reda veličine razlika između konfiguracija. K-fold validacija nad trening i validacionim skupom zajedno (oko 37.000 naslova) dala bi pouzdaniji izbor i manje prilagođavanje jednom skupu.
- **Ranija provera na više seed-ova**, pre zaključaka iz pojedinačnih runova.
- **Brisanje reči i iz niza karaktera tokom treniranja** ("span dropout"), jer grana karaktera trenutno vidi i reči koje je word dropout sakrio (vidi [arhitekturu](architecture.md)).
- **Kalibracija svakog člana pre spajanja** modela različitog tipa u ansambl.

## 10. Reprodukcija

```bash
python -m kp_classifier.models.baseline --eval-test
for seed in 42 1 2; do
  python -m kp_classifier.train --config configs/hybrid_final.yaml --name "maxmean_ls01_s${seed}" --set "seed=${seed}"
done
python -m kp_classifier.ensemble --name ens_mm_ls_cal \
  --models maxmean_ls01_s42 maxmean_ls01_s1 maxmean_ls01_s2 --calibrate --eval-test
python -m kp_classifier.evaluate mcnemar baseline_tfidf ens_mm_ls_cal --split test
python -m kp_classifier.evaluate compare --split test
```

Za svaki run, krive učenja (`learning_curves.png`), istorija po epohama (`history.csv`) i sve metrike nalaze se u `reports/runs/<ime>/`. Na Apple MPS-u treniranje nije bit-za-bit ponovljivo, pa se brojke mogu razlikovati u nekoliko primera.
