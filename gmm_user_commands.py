import time
from contextlib import closing

import gmm_model as gmm
import record

THRESHOLD_FILENAME = 'threshold.txt'
THRESHOLD_DEFAULT = 22.0
TEMP_FILENAME = 'temp.wav'

def _read_threshold():
    try:
        with closing(open(THRESHOLD_FILENAME, 'r')) as tsh_file:
            threshold = float(tsh_file.readline()) # pylint: disable=no-member
    except FileNotFoundError:
        threshold = THRESHOLD_DEFAULT
    return threshold

def _record_voice(filename=None, length_sec=3):
    frames = record.record_frames(length_sec)
    if filename is None:
        record.write_frames(frames)
    else:
        record.write_frames(frames, filename)

def write_threshold():
    try:
        threshold = float(input("Podaj nowy próg:\n>"))
        with closing(open(THRESHOLD_FILENAME, 'w')) as tsh_file:
            tsh_file.write(str(threshold)) # pylint: disable=no-member
        print("Zapisano nowy próg!")
    except:
        print("Błąd. Przerywanie.")

def create_model():
    print("Tworzenie modelu!")
    gm = gmm.train_model(0)
    print("Zapisywanie modelu!")
    gmm.write_model(gm)
    print("Zakończono")

def test_run():
    print("Wczytywanie modelu!")
    gm = gmm.read_model()
    threshold = _read_threshold()
    print("Przeprowadznie testów")
    test_res = gmm.test_model(gm)
    for test_re in test_res:
        is_ok_cap = "Dobra" if test_re.is_ok else "Zła"
        verdict_cap = "Pod progiem" if test_re.score < threshold else "Ponad progiem"
        print("{} próbka {} - wynik {} - wedykt - {}".format(is_ok_cap, test_re.filename, test_re.score, verdict_cap))

def interactive_test():
    print("Wczytywanie modelu!")
    gm = gmm.read_model()
    threshold = _read_threshold()
    try:
        length_raw = input("Podaj długość próbki (w sekundach!) [15]")
        length = 15 if length_raw == '' else float(length_raw)
    except:
        print("Błąd. Użycie długości 15s...")
        length = 15
    print("Nagrywanie rozpocznie się za 3 sek.")
    time.sleep(3)
    _record_voice(TEMP_FILENAME, length)
    print("Analizowanie próbki...")
    score = gmm.sample_test(gm, TEMP_FILENAME)
    print("Wynik: {}".format(score))
    print("Próg: {}".format(threshold))
    if score > threshold:
        print("Zaakceptowano")
    else:
        print("Odrzucono")
