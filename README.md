# Klasifikacija oglasa po grupama na osnovu naziva

Neuronska mreža napravljena i obučena od nule (bez pretreniranih modela i embedding-a) koja naslov oglasa svrstava u jednu od 10 grupa. Konačni model je ansambl tri hibridne konvolucione mreže sa granom reči i granom karaktera, uz kalibraciju verovatnoća.

## Rezultati na test skupu (4.150 naslova)

| model | accuracy | macro F1 | greške | ECE |
|---|---|---|---|---|
| **ansambl hibridnih CNN-ova (konačni model)** | **98,99%** [98,67 – 99,28] | 0,9892 | 42 | **0,0033** |
| TF-IDF + logistička regresija (baseline) | 99,16% [98,89 – 99,42] | 0,9912 | 35 | 0,0058 |
| trivijalni model (uvek najčešća klasa) | 11,90% | 0,0213 | 3.656 | – |

U zagradama su 95% bootstrap intervali poverenja. Razlika između ansambla i baseline-a **nije statistički značajna** (McNemar, p = 0,26). Detaljno tumačenje, uključujući zašto je prednost mreže izmerena na validacionom skupu nestala na testu, nalazi se u [izveštaju sa treniranja](reports/training_report.md).

Ključni nalazi:
- **grana karaktera** smanjuje broj grešaka za 23% na originalnim naslovima i za 61% na naslovima sa greškama u kucanju (u odnosu na istu mrežu samo sa rečima),
- **label smoothing** je bio presudan: bez njega mreža je bila previše samouverena i lošija od baseline-a,
- uz prag sigurnosti 0,99, model automatski klasifikuje ~96% oglasa sa tačnošću od 99,9%.

## Instalacija

Testirano sa Python 3.14.2 na macOS (Apple M1). Potreban je Python 3.10 ili noviji.

~~~bash
git clone https://github.com/Punosevacx2/kp-ad-classifier.git
cd kp-ad-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
~~~

Tačne verzije paketa korišćene pri razvoju nalaze se u `requirements-lock.txt`.

## Klasifikacija naslova gotovim modelom

Konačni model je u repozitorijumu (`models/ens_mm_ls_cal`), pa za predikciju nisu potrebni podaci ni baza.

~~~bash
python -m kp_classifier.predict "iPhone 13 Pro 128GB" "Ugaona brusilica Makita 125mm" "Krevet na sprat"
~~~

~~~
iPhone 13 Pro 128GB
  -> Apple iPhone (489)  0.999
     2. ...
~~~

Opcije:

| opcija | opis |
|---|---|
| `--json` | izlaz u JSON formatu, sa `group_id`, nazivom grupe, sigurnošću i top-k grupama |
| `--top-k N` | broj najverovatnijih grupa u izlazu (podrazumevano 3) |
| `--threshold T` | naslovi sa sigurnošću ispod `T` označavaju se za ručnu proveru (podrazumevano 0,9) |
| `--file putanja` | ulaz iz `.txt` (jedan naslov po redu) ili `.csv` sa kolonom `name` |
| `--output putanja.csv` | rezultati u CSV fajl |
| `--model ime` | drugi model iz foldera `models/`, npr. `baseline_tfidf` (ako je istreniran lokalno) |

Iz Python koda:

~~~python
from kp_classifier.predict import Predictor

predictor = Predictor("models/ens_mm_ls_cal")
predictor.predict(["Plišani meda 40cm"])
~~~

## Ponovno treniranje od nule

Za export podataka potreban je pristup MySQL bazi. Napravi `.env` po uzoru na `.env.example` i upiši pristupne podatke. Podaci se ne nalaze u repozitorijumu.

Ceo tok jednom komandom (oko 30 minuta na Apple M1):

~~~bash
./scripts/reproduce.sh
~~~

Isto, korak po korak:

| korak | komanda | izlaz |
|---|---|---|
| 1. export iz baze | `python -m kp_classifier.data.export_mysql` | `data/raw/ml_dataset.parquet` |
| 2. čišćenje i podela 80/10/10 | `python -m kp_classifier.data.split` | `data/processed/{train,val,test}.parquet`, `label_map.json` |
| 3. statistika vokabulara | `python -m kp_classifier.data.tokenizer` | `data/processed/vocab.json` |
| 4. baseline | `python -m kp_classifier.models.baseline --eval-test` | `models/baseline_tfidf/` |
| 5. tri mreže (seed 42, 1, 2) | `python -m kp_classifier.train --config configs/hybrid_final.yaml --name maxmean_ls01_s42 --set seed=42` | `models/maxmean_ls01_s*/` |
| 6. ansambl i kalibracija | `python -m kp_classifier.ensemble --name ens_mm_ls_cal --models maxmean_ls01_s42 maxmean_ls01_s1 maxmean_ls01_s2 --calibrate --eval-test` | `models/ens_mm_ls_cal/` |
| 7. poređenje | `python -m kp_classifier.evaluate compare --split test` | tabela svih modela |

Svaki run snima metrike, konfuzionu matricu, dijagram kalibracije, krivu rizik–pokrivenost, krive učenja i listu grešaka u `reports/runs/<ime>/`.

Bilo koji parametar iz configa može se promeniti bez izmene fajla:

~~~bash
python -m kp_classifier.train --config configs/hybrid_final.yaml --name eksperiment --set model.use_chars=false --set training.learning_rate=0.001
~~~

Uređaj se bira automatski (CUDA, Apple MPS ili CPU). Na MPS-u rezultati nisu bit-za-bit ponovljivi, pa se brojke mogu razlikovati u nekoliko primera između pokretanja.

## Testovi

~~~bash
python -m pytest -q tests
~~~

Testovi pokrivaju normalizaciju (uključujući prepoznavanje telefona i kataloških brojeva), podelu podataka bez curenja, tokenizaciju, arhitekturu (npr. da dopuna `<pad>` ne menja izlaz), treniranje, metrike, ansambl, kalibraciju, testove robusnosti i predikciju.

## Struktura projekta

~~~
kp-ad-classifier/
├── configs/
│   ├── baseline_tfidf.yaml        parametri baseline-a
│   ├── textcnn.yaml               početna konfiguracija mreže
│   └── hybrid_final.yaml          konfiguracija konačnog modela
├── src/kp_classifier/
│   ├── data/
│   │   ├── export_mysql.py        export tabele ml_dataset
│   │   ├── preprocess.py          normalizacija teksta
│   │   ├── split.py               čišćenje, deduplikacija, stratifikovana podela
│   │   ├── tokenizer.py           vokabular reči i karaktera
│   │   └── dataset.py             PyTorch DataLoader
│   ├── models/
│   │   ├── baseline.py            TF-IDF + logistička regresija
│   │   └── textcnn.py             hibridni CNN
│   ├── train.py                   treniranje mreže
│   ├── ensemble.py                ansambl i temperature scaling
│   ├── evaluate.py                metrike, intervali poverenja, McNemar test
│   ├── robustness.py              testovi robusnosti
│   └── predict.py                 predikcija za nove naslove
├── models/ens_mm_ls_cal/          konačni model (ansambl)
├── models/maxmean_ls01_s*/        tri člana ansambla
├── notebooks/01_eda.ipynb         eksplorativna analiza podataka
├── reports/                       izveštaji, grafici i rezultati svih eksperimenata
├── scripts/reproduce.sh           ceo tok od exporta do evaluacije
└── tests/                         automatski testovi
~~~

## Dokumentacija

| dokument | sadržaj |
|---|---|
| [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) | eksplorativna analiza podataka sa zaključcima |
| [reports/data_preparation.md](reports/data_preparation.md) | priprema trening podataka |
| [reports/architecture.md](reports/architecture.md) | arhitektura modela sa dijagramom |
| [reports/metrics.md](reports/metrics.md) | objašnjenje metrika |
| [reports/training_report.md](reports/training_report.md) | izveštaj sa treniranja i rezultati svih eksperimenata |
| [reports/error_analysis.md](reports/error_analysis.md) | analiza grešaka i robusnosti |

## Ograničenja

- Model uvek bira jednu od 10 grupa i nema opciju "nijedna od ponuđenih". Naslovi koji ne pripadaju nijednoj grupi (npr. "Stolovi za ugostiteljstvo") dobijaju neku od 10 grupa, ponekad i sa visokom sigurnošću.
- Kalibracija važi u proseku: pojedinačne greške i dalje mogu imati visoku sigurnost.
- Deo grešaka potiče iz pogrešnih labela u podacima, pa tačnost od 100% nije dostižna.
