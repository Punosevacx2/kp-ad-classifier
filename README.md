# Klasifikacija oglasa po grupama na osnovu naziva

Rešenje zadatka za kandidate (AI): model koji naslov oglasa svrstava u jednu od 10 grupa. Model je neuronska mreža **projektovana i obučena od nule**, bez pretreniranih modela i embedding-a. Konačni model je ansambl tri hibridne konvolucione mreže sa granom reči i granom karaktera, uz kalibraciju verovatnoća.

Autor: Miloš Punosevac

## Gde se nalazi šta iz zadatka

| traženo u zadatku | gde se nalazi |
|---|---|
| model sa uputstvom za pokretanje | `models/ens_mm_ls_cal/` (konačni model) i ovaj README, odeljak [Klasifikacija naslova](#klasifikacija-naslova-gotovim-modelom) |
| kod za obučavanje i ostali kod | `src/kp_classifier/` (opis svakog modula u [strukturi projekta](#struktura-projekta)), testovi u `tests/` |
| objašnjenje metrike | [reports/metrics.md](reports/metrics.md) |
| izveštaj sa treniranja, primenjene metrike i način obučavanja | [reports/training_report.md](reports/training_report.md) |
| način pripreme trening podataka | [reports/data_preparation.md](reports/data_preparation.md) i [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) |
| arhitektura modela (dijagrami, opis) | [reports/architecture.md](reports/architecture.md) |
| dodatni uvidi i unapređenja | [reports/error_analysis.md](reports/error_analysis.md): analiza grešaka, testovi robusnosti, preporuke za upotrebu; kalibracija i kriva rizik–pokrivenost u [metrikama](reports/metrics.md) |

## Rezultati na test skupu (4.150 naslova)

| model | accuracy | macro F1 | greške | ECE |
|---|---|---|---|---|
| **ansambl hibridnih CNN-ova (konačni model)** | **98,99%** [98,67 – 99,28] | 0,9892 | 42 | **0,0033** |
| TF-IDF + logistička regresija (referentni model) | 99,16% [98,89 – 99,42] | 0,9912 | 35 | 0,0058 |
| trivijalni model (uvek najčešća klasa) | 11,90% | 0,0213 | 3.656 | – |

U zagradama su 95% bootstrap intervali poverenja. Razlika između ansambla i referentnog modela **nije statistički značajna** (McNemar, p = 0,26). Zašto je prednost mreže izmerena na validacionom skupu nestala na test skupu, objašnjeno je u [izveštaju sa treniranja](reports/training_report.md#7-rezultat-na-test-skupu).

Ključni nalazi:
- **grana karaktera** smanjuje broj grešaka za 23% na originalnim naslovima i za 61% na naslovima sa greškama u kucanju (u odnosu na istu mrežu samo sa rečima),
- **label smoothing** je bio presudan: bez njega mreža je bila previše samouverena i lošija od referentnog modela,
- uz prag sigurnosti 0,99, model automatski klasifikuje ~96% oglasa sa tačnošću od 99,9%.

## Instalacija

Testirano sa Python 3.14.2 na macOS (Apple M1). Potreban je Python 3.10 ili noviji.

```bash
git clone https://github.com/Punosevacx2/kp-ad-classifier.git
cd kp-ad-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` instalira paket `kp_classifier`, pa sve komande ispod rade iz korena projekta. Tačne verzije paketa korišćene pri razvoju nalaze se u `requirements-lock.txt`.

Provera instalacije:

```bash
python -m pytest -q tests
```

Očekivano: `109 passed`.

## Klasifikacija naslova gotovim modelom

Konačni model je u repozitorijumu, pa za predikciju **nisu potrebni podaci ni pristup bazi**.

```bash
python -m kp_classifier.predict "iPhone 13 Pro 128GB" "Ugaona brusilica Makita 125mm" "Krevet na sprat"
```

```
iPhone 13 Pro 128GB
  -> Apple iPhone (489)  0.999
     2. ...
```

Za svaki naslov ispisuje se grupa (naziv i `group_id`), sigurnost modela i sledeće najverovatnije grupe. Naslovi sa sigurnošću ispod praga označeni su sa `[PROVERITI]`.

| opcija | opis |
|---|---|
| `--json` | izlaz u JSON formatu, sa `group_id`, nazivom grupe, sigurnošću i top-k grupama |
| `--top-k N` | broj najverovatnijih grupa u izlazu (podrazumevano 3) |
| `--threshold T` | naslovi sa sigurnošću ispod `T` označavaju se za ručnu proveru (podrazumevano 0,9) |
| `--file putanja` | ulaz iz `.txt` (jedan naslov po redu) ili `.csv` sa kolonom `name` |
| `--output putanja.csv` | rezultati u CSV fajl |
| `--model ime` | drugi model iz foldera `models/`, npr. `baseline_tfidf` (posle lokalnog treniranja) |

Iz Python koda:

```python
from kp_classifier.predict import Predictor

predictor = Predictor("models/ens_mm_ls_cal")
predictor.predict(["Plišani meda 40cm"])
```

## Pristup podacima

Podaci se ne nalaze u repozitorijumu. Za sve korake ispod potreban je pristup MySQL bazi iz zadatka:

```bash
cp .env.example .env
```

pa u `.env` upisati `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD` i `MYSQL_DATABASE`. Fajl `.env` je u `.gitignore`.

```bash
python -m kp_classifier.data.export_mysql
python -m kp_classifier.data.split
```

Podela je deterministička: isti podaci uvek daju iste trening, validacioni i test skup.

## Provera rezultata bez ponovnog treniranja

Posle exporta i podele, rezultati konačnog modela na test skupu mogu se proveriti za par minuta, sa mrežama iz repozitorijuma:

```bash
python -m kp_classifier.ensemble --name ens_mm_ls_cal \
  --models maxmean_ls01_s42 maxmean_ls01_s1 maxmean_ls01_s2 --calibrate --eval-test
python -m kp_classifier.models.baseline --eval-test
python -m kp_classifier.evaluate mcnemar baseline_tfidf ens_mm_ls_cal --split test
python -m kp_classifier.evaluate compare --split test
```

Prva komanda ponovo bira temperaturu kalibracije na validacionom skupu (dobija se ista vrednost, T = 0,508) i ocenjuje ansambl na test skupu. Druga trenira referentni model (oko minut).

## Ponovno treniranje od nule

Ceo tok jednom komandom (oko 30 minuta na Apple M1):

```bash
./scripts/reproduce.sh
```

Isto, korak po korak:

| korak | komanda | izlaz |
|---|---|---|
| 1. export iz baze | `python -m kp_classifier.data.export_mysql` | `data/raw/ml_dataset.parquet` |
| 2. čišćenje i podela 80/10/10 | `python -m kp_classifier.data.split` | `data/processed/{train,val,test}.parquet`, `label_map.json` |
| 3. statistika vokabulara | `python -m kp_classifier.data.tokenizer` | `data/processed/vocab.json` |
| 4. referentni model | `python -m kp_classifier.models.baseline --eval-test` | `models/baseline_tfidf/` |
| 5. tri mreže (seed 42, 1, 2) | `python -m kp_classifier.train --config configs/hybrid_final.yaml --name maxmean_ls01_s42 --set seed=42` | `models/maxmean_ls01_s*/` |
| 6. ansambl i kalibracija | `python -m kp_classifier.ensemble --name ens_mm_ls_cal --models maxmean_ls01_s42 maxmean_ls01_s1 maxmean_ls01_s2 --calibrate --eval-test` | `models/ens_mm_ls_cal/` |
| 7. testovi robusnosti | `python -m kp_classifier.robustness --models baseline_tfidf ens_mm_ls_cal --split test --out-dir reports/robustness_test` | `reports/robustness_test/` |
| 8. poređenje | `python -m kp_classifier.evaluate compare --split test` | tabela svih modela |

Svaki run snima metrike, konfuzionu matricu, dijagram kalibracije, krivu rizik–pokrivenost, krive učenja i listu grešaka u `reports/runs/<ime>/`.

Bilo koji parametar iz configa može se promeniti bez izmene fajla:

```bash
python -m kp_classifier.train --config configs/hybrid_final.yaml --name eksperiment \
  --set model.use_chars=false --set training.learning_rate=0.001
```

Uređaj se bira automatski (CUDA, Apple MPS ili CPU). Na MPS-u rezultati nisu bit-za-bit ponovljivi, pa se brojke mogu razlikovati u nekoliko primera između pokretanja.

## Eksplorativna analiza podataka

```bash
pip install jupyter
jupyter notebook notebooks/01_eda.ipynb
```

Notebook čita `data/raw/ml_dataset.parquet` (posle exporta), a grafike snima u `reports/figures/`. Zaključci analize su na kraju notebook-a.

## Struktura projekta

```
kp-ad-classifier/
├── configs/
│   ├── baseline_tfidf.yaml        parametri referentnog modela
│   ├── textcnn.yaml               početna konfiguracija mreže
│   └── hybrid_final.yaml          konfiguracija konačnog modela
├── src/kp_classifier/
│   ├── data/
│   │   ├── export_mysql.py        export tabele ml_dataset iz baze
│   │   ├── preprocess.py          normalizacija teksta (ćirilica, dijakritici, <phone>, <id>)
│   │   ├── split.py               čišćenje, deduplikacija, stratifikovana podela
│   │   ├── tokenizer.py           vokabular reči i karaktera
│   │   └── dataset.py             PyTorch DataLoader
│   ├── models/
│   │   ├── baseline.py            TF-IDF + logistička regresija
│   │   └── textcnn.py             arhitektura hibridne mreže
│   ├── train.py                   treniranje mreže
│   ├── ensemble.py                ansambl i temperature scaling
│   ├── evaluate.py                metrike, intervali poverenja, McNemar test
│   ├── robustness.py              testovi robusnosti
│   └── predict.py                 predikcija za nove naslove
├── models/
│   ├── ens_mm_ls_cal/             konačni model (ansambl, temperatura)
│   └── maxmean_ls01_s{42,1,2}/    tri člana ansambla (težine, vokabular, config)
├── notebooks/01_eda.ipynb         eksplorativna analiza podataka
├── reports/
│   ├── *.md                       izveštaji (vidi tabelu na vrhu)
│   ├── runs/<model>/              metrike, grafici i greške svakog eksperimenta
│   ├── robustness/                testovi robusnosti, validacioni skup
│   ├── robustness_test/           testovi robusnosti, test skup
│   └── figures/                   grafici iz analize podataka
├── scripts/reproduce.sh           ceo tok od exporta do evaluacije
├── tests/                         109 automatskih testova
├── .env.example                   šablon za pristup bazi
├── requirements.txt               potrebni paketi
├── requirements-lock.txt          tačne verzije paketa
└── pyproject.toml                 definicija paketa kp_classifier
```

## Testovi

```bash
python -m pytest -q tests
```

Testovi pokrivaju normalizaciju (uključujući prepoznavanje telefona i kataloških brojeva), podelu podataka bez curenja, tokenizaciju, arhitekturu (npr. da dopuna `<pad>` ne menja izlaz), treniranje, metrike, ansambl, kalibraciju, testove robusnosti i predikciju. Ne zahtevaju pristup bazi.

## Ograničenja

- Model uvek bira jednu od 10 grupa i nema opciju "nijedna od ponuđenih". Naslovi koji ne pripadaju nijednoj grupi (npr. "Stolovi za ugostiteljstvo") dobijaju neku od 10 grupa, ponekad i sa visokom sigurnošću.
- Kalibracija važi u proseku: pojedinačne greške i dalje mogu imati visoku sigurnost.
- Deo grešaka potiče iz pogrešnih labela u podacima, pa tačnost od 100% nije dostižna.
