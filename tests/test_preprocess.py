import pytest

from kp_classifier.data.preprocess import normalize_text, replace_phones


@pytest.mark.parametrize("text", [
    "065/280-20-20",
    "0613004930",
    "064 123 4567",
    "063-123-456",
    "011/123-45-67",
    "018 123 456",
    "+381 65 2802020",
    "+381652802020",
    "00381 63 123 4567",
    "+381 (0) 65 280 20 20",
])
def test_phone_is_replaced(text):
    assert replace_phones(text).strip() == "<phone>"


@pytest.mark.parametrize("text", [
    "iPhone 15 Pro 128GB",
    "Krevet 160x200",
    "Plac 3937m2",
    "Crvena zvezda prvak 1969 1969",
    "Pokemon kartica 022/067 RR",
    "IMT 507 zupčanik 507.01.119",
    "Nano 0.1 zastitno staklo",
    "Poljoprivredno zemljište, Žabalj, 42196m2, 532632",
    "Brusilica 2200W AG220018",
    "iPhone 12 128gb - 9,9/10",
    "Bosch Ugaona brusilica 125mm 720W - GWS 7-125 (0601388108)",
    "Bosch PSM 80 A, 0603354000 Multi-brusilica",
    "Bosch PEX 220 A šlajferica - ekscentar brusilica, 060337802",
    "BOSCH 061599405U GRL 300 HV Professional Rotary Laser",
    "Ekscentar brusilica Bosch GET 55-125 (0601257000)",
    "ETA 0168 90000 Vista Hot air oven 1300 W 4.3 l black",
    "ETA 0133 90020 Electric Baking Cool 475W 4.3L Black/Silver",
])
def test_no_false_positive(text):
    assert "<phone>" not in replace_phones(text)


@pytest.mark.parametrize("raw, expected", [
    ("OTKUP APPLE IPHONE SE 2022 kupujem   065/280-20-20", "otkup apple iphone se 2022 kupujem <phone>"),
    ("Otkup iPhone 15proMax 0613004930 (ne radi donji zvucnik)", "otkup iphone 15promax <phone> ne radi donji zvucnik"),
    ("Plac u Bockama 2101m2! Sifra: 528492", "plac u bockama 2101m2 <id>"),
    ("Кревети од масива буква", "kreveti od masiva bukva"),
    ("Француски лежај", "francuski lezaj"),
    ("Daewoo vibraciona brusilica 113х225, 380W", "daewoo vibraciona brusilica 113x225 380w"),
    ("Dušek za bračni krevet", "dusek za bracni krevet"),
    ("Đubrivo i ĐAČKI sto", "djubrivo i djacki sto"),
    ("Iphone 16 Pro - NA STANJU!", "iphone 16 pro na stanju"),
    ("(Rezervisan)iPhone 6s", "rezervisan iphone 6s"),
    ("snake_case__tekst", "snake case tekst"),
    ("Otkup mobilnih telefona Bg- Ns/0603333828", "otkup mobilnih telefona bg ns <phone>"),
    ("OTKUP APPLE IPHONE 13 Starlight/064/954-23-97", "otkup apple iphone 13 starlight <phone>"),
    ("OTKUP IPHONE 14 Pro 128/064/954-23-97", "otkup iphone 14 pro 128 <phone>"),
    ("Polirka Bosch GPO 14 CE (0601389000)", "polirka bosch gpo 14 ce 0601389000"),
    ("Ugaona brusilica GWS 750 S Professional 0601394120", "ugaona brusilica gws 750 s professional 0601394120"),
    ("Bosch brusilica, zvati 060 123 4567", "bosch brusilica zvati <phone>"),
    ("Dupont upaljace KUPUJEM 0611911355", "dupont upaljace kupujem <phone>"),
    ("ETA friteza, pozovite 064/822-21-40", "eta friteza pozovite <phone>"),
    ("Petrovaradin, 1652m2 /  Šifra 535449", "petrovaradin 1652m2 <id>"),
    ("Brusilica TC-ID 750 E", "brusilica tc id 750 e"),
    ("Udarna bušilica TC - ID 720/1 E KIT", "udarna busilica tc id 720 1 e kit"),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected