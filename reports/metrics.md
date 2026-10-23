# Metrike

Ovaj dokument objašnjava kojim metrikama je merena preciznost modela, zašto baš njima, i kako su modeli poređeni. Sve metrike računa modul `src/kp_classifier/evaluate.py`, identično za svaki model.

## 1. Priroda problema

- **10 klasa**, svaki naslov pripada tačno jednoj grupi,
- posle čišćenja klase **nisu iste veličine**: najveća (Samsung futrole, 4.935) je 1,67 puta veća od najmanje (Roštilji, 2.954),
- validacioni i test skup imaju po **4.150 naslova**, pa svaka greška menja tačnost za 0,024 procentna poena,
- u podacima postoje **pogrešne labele**, pa 100% tačnosti nije dostižno.

## 2. Glavna metrika: macro F1

Za svaku grupu računaju se tri veličine. Primer za grupu Plišane igračke:

| veličina | formula | pitanje na koje odgovara |
|---|---|---|
| **preciznost** (precision) | TP / (TP + FP) | od naslova koje je model svrstao u Plišane igračke, koliko ih stvarno jeste plišana igračka? |
| **odziv** (recall) | TP / (TP + FN) | od svih stvarnih plišanih igračaka, koliko ih je model pronašao? |
| **F1** | 2 · P · R / (P + R) | harmonijska sredina preciznosti i odziva |

TP su tačno prepoznati primeri grupe, FP primeri drugih grupa pogrešno svrstani u nju, a FN primeri grupe svrstani negde drugde.

**Macro F1** je prost prosek F1 vrednosti svih 10 grupa:

```
macro F1 = (F1₁ + F1₂ + … + F1₁₀) / 10
```

**Zašto je glavna:** svaka grupa ima istu težinu, bez obzira na veličinu. Model koji bi loše radio na najmanjoj grupi (Roštilji) ne može to da sakrije dobrim rezultatom na velikim grupama, kao što bi mogao kod same tačnosti. F1 takođe kažnjava i lažne pozitive i lažne negative.

## 3. Tačnost (accuracy)

```
accuracy = broj tačnih predikcija / ukupan broj naslova
```

Najintuitivnija metrika i zato uvek prijavljena. Uz blagu neravnotežu klasa, tačnost i macro F1 su ovde vrlo bliski (98,99% i 0,9892 za konačni model na testu), ali macro F1 ostaje pouzdaniji kriterijum.

**Trivijalni model** koji uvek bira najveću grupu ima tačnost 11,9% i macro F1 0,021. To je donja granica od koje se meri svaki model.

## 4. Metrike po grupama i konfuziona matrica

Za svaku grupu prijavljuju se preciznost, odziv i F1 (`*_per_class.csv`), a za svaki par grupa broj zamena (`*_top_confusions.csv` i `*_confusion_matrix.png`).

U konfuzionoj matrici red je stvarna grupa, a kolona predviđena grupa. Broj u polju je broj naslova, a boja procenat stvarne grupe, pa se male i velike grupe mogu porediti.

Primer tumačenja: grupa Plišane igračke ima **najnižu preciznost** kod svih modela. Kada naslov nema jasan signal (ime lika, brend, ime mesta), modeli ga češće svrstavaju u Plišane igračke, pa ta grupa "prima" greške iz drugih grupa. Ovaj obrazac se vidi samo u metrikama po grupama, ne u ukupnoj tačnosti.

## 5. Log loss

```
log loss = − (1/N) · Σ log p(tačna grupa)
```

Prosečan negativni logaritam verovatnoće koju je model dao **tačnoj** grupi. Ne meri samo da li je odluka tačna, nego i koliko je model u nju bio siguran: tačan odgovor sa verovatnoćom 0,51 kažnjava se više nego sa 0,99, a pogrešan odgovor sa velikom sigurnošću kažnjava se veoma strogo.

**Gde je korišćen:**
- **izbor hiperparametara kod izjednačenih modela:** kod baseline-a su vrednosti `C` od 3 do 100 imale praktično isti macro F1 (razlika manja od jedne greške), ali log loss od 0,061 do 0,034, pa je odlučio log loss,
- **izbor epohe pri treniranju mreže** (rano zaustavljanje),
- **izbor temperature** pri kalibraciji.

Prednost u odnosu na macro F1 je što ne zavisi od nekoliko graničnih primera, pa je manje bučan na skupu od 4.150 naslova.

Napomena: modeli obučeni sa label smoothing-om namerno daju umerene verovatnoće, pa njihov log loss pre kalibracije nije uporediv sa ostalim modelima.

## 6. Top-2 tačnost

Udeo naslova kod kojih je tačna grupa među dve najverovatnije. Praktično značenje: sistem koji korisniku pri postavljanju oglasa **predlaže dve grupe** bio bi tačan u 99,5% slučajeva (konačni model, test skup).

## 7. Kalibracija: ECE

Model je **dobro kalibrisan** ako njegova sigurnost odgovara stvarnoj tačnosti: od svih naslova za koje kaže "95% siguran", tačan je u oko 95% slučajeva.

**ECE** (Expected Calibration Error) se računa tako što se naslovi podele u 15 intervala po sigurnosti, pa se za svaki interval meri razlika između prosečne sigurnosti i stvarne tačnosti:

```
ECE = Σ (n_b / N) · | tačnost_b − prosečna sigurnost_b |
```

Vizuelno: dijagram kalibracije (`*_reliability.png`), na kome savršeno kalibrisan model leži na dijagonali.

| model | ECE (test) |
|---|---|
| ansambl pre kalibracije (validacioni skup) | 0,104 |
| **ansambl posle kalibracije** | **0,003** |
| baseline | 0,006 |

**Važno:** kalibracija ne govori ništa o tome koliko je model koristan. Trivijalni model koji uvek bira najveću grupu ima ECE 0,0002, jer kaže "12% siguran" i tačan je u 12% slučajeva. Zato se ECE uvek posmatra uz tačnost.

## 8. Kriva rizik–pokrivenost

Odgovara na praktično pitanje: **koliki deo oglasa model sme da klasifikuje automatski, uz koju tačnost?**

Naslovi se sortiraju po sigurnosti modela, pa se za svaki prag meri tačnost na naslovima iznad praga. Ostali se šalju na ručnu proveru ili se korisniku nude dve grupe.

Test skup:

| ciljna tačnost | ansambl: pokrivenost (prag) | baseline: pokrivenost (prag) |
|---|---|---|
| 99,0% | 100,0% | 100,0% |
| 99,5% | 99,1% (0,751) | 98,8% (0,599) |
| 99,9% | 95,8% (0,990) | 95,9% (0,938) |

Tumačenje: oba modela mogu automatski da klasifikuju oko **96% oglasa sa greškom od jedne na hiljadu**. Ova metrika je smislena samo za dobro kalibrisan model, jer se prag bira na osnovu sigurnosti.

## 9. Statistička pouzdanost

### Zašto je potrebna

Isti baseline model imao je 98,65% tačnosti na jednoj nasumičnoj podeli i 99,01% na drugoj, a 98,65% na validacionom i 99,16% na test skupu. Sa 4.150 primera, razlike od nekoliko desetina procentnog poena mogu biti posledica samo toga koji su naslovi završili u kom skupu. Sličan efekat ima i seed pri treniranju mreže: ista konfiguracija sa tri seed-a dala je 72, 74 i 88 grešaka.

### Bootstrap interval poverenja

Skup se 1.000 puta nasumično uzorkuje sa ponavljanjem, metrika se računa na svakom uzorku, a 2,5. i 97,5. percentil daju **95% interval poverenja**. Prijavljuje se za tačnost i macro F1:

```
accuracy: 0.9899  [0.9867, 0.9928]
```

Interval je širok oko ±0,3 procentna poena. Ako se intervali dva modela preklapaju, nije moguće zaključiti koji je bolji samo na osnovu njih.

### McNemar test

Poredi dva modela **na istim primerima**. Gleda samo naslove na kojima se modeli ne slažu:

- `b`: model A tačan, model B pogrešan,
- `c`: model A pogrešan, model B tačan.

Ako su modeli jednako dobri, `b` i `c` bi trebalo da budu slični. Egzaktni binomni test daje verovatnoću (p-vrednost) da tolika razlika nastane slučajno. Ispod 0,05 razlika se smatra statistički značajnom.

Test je osetljiviji od poređenja dva intervala poverenja, jer zanemaruje hiljade naslova na kojima su oba modela tačna. Konačni rezultat:

| skup | samo baseline tačan | samo ansambl tačan | p |
|---|---|---|---|
| validacioni | 4 | 24 | 0,0002 |
| test | 18 | 11 | 0,26 |

### Višestruka poređenja

Kada se proba mnogo konfiguracija i izabere najbolja na istom skupu, pobednik je delom pobednik i zbog slučajnosti. **Bonferroni korekcija** množi p-vrednost brojem poređenja. Ali ni ona ne štiti od toga da su sve konfiguracije podešavane gledajući isti validacioni skup. Upravo zato postoji test skup.

## 10. Protokol evaluacije

| skup | koristi se za |
|---|---|
| trening | učenje težina i vokabulara |
| validacioni | sve odluke: hiperparametri, arhitektura, izbor epohe, temperatura |
| test | **jedna** konačna ocena, posle zamrzavanja modela |

Svaka konfiguracija mreže trenirana je sa tri seed-a, a zaključci su donošeni na osnovu proseka i doslednosti preko seed-ova, ne na osnovu jednog runa.

## 11. Konačni rezultati (test skup, 4.150 naslova)

| metrika | ansambl (konačni model) | baseline |
|---|---|---|
| accuracy | 98,99% [98,67 – 99,28] | 99,16% [98,89 – 99,42] |
| macro F1 | 0,9892 [0,9859 – 0,9923] | 0,9912 [0,9883 – 0,9939] |
| greške | 42 | 35 |
| log loss | 0,0382 | 0,0382 |
| top-2 accuracy | 99,52% | 99,49% |
| ECE | 0,0033 | 0,0058 |
| pokrivenost uz 99,9% tačnosti | 95,8% | 95,9% |

Razlika u tačnosti nije statistički značajna (McNemar, p = 0,26).

## 12. Gde se nalaze rezultati

Za svaki model i skup, u `reports/runs/<model>/`:

| fajl | sadržaj |
|---|---|
| `*_metrics.json` | sve metrike, intervali poverenja i pokrivenost |
| `*_per_class.csv` | preciznost, odziv, F1 i broj primera po grupi |
| `*_top_confusions.csv` | najčešće zamene grupa |
| `*_errors.csv` | sve greške, sortirane po sigurnosti modela |
| `*_confusion_matrix.png` | konfuziona matrica |
| `*_reliability.png` | dijagram kalibracije |
| `*_risk_coverage.png` | kriva rizik–pokrivenost |

Poređenje svih modela i McNemar test:

```bash
python -m kp_classifier.evaluate compare --split test
python -m kp_classifier.evaluate mcnemar baseline_tfidf ens_mm_ls_cal --split test
```
