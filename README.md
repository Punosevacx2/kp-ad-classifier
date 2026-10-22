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

```bash
git clone https://github.com/Punosevacx2/kp-ad-classifier.git
cd kp-ad-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Tačne verzije paketa korišćene pri razvoju nalaze se u `requirements-lock.txt`.

## Klasifikacija naslova gotovim modelom

Konačni model je u repozitorijumu (`models/ens_mm_ls_cal`), pa za predikciju nisu potrebni podaci ni baza.

```bash
python -m kp_classifier.predict "iPhone 13 Pro 128GB" "Ugaona brusilica Makita 125mm" "Krevet na sprat"
```