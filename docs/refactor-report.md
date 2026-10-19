# Raport przebudowy API

Praca została wykonana na `dev_max_integrated`, od commita
`27d01deb58852a0991b8c39ec681e637e735021e`. Na początku odświeżyłem tę gałąź
z `origin` i sprawdziłem czyste drzewo robocze. Zmiany nie zostały zacommitowane.

## Kolejność prac i uzasadnienie

1. Spisałem kryteria akceptacji: bezpośrednie importy, brak rejestrów i adapterów
   starego API, rozdzielenie wrappera/buildera/schema, niezależność buildów,
   zachowanie obliczeń i zapisu, wspólne konstruktory dla Pythona i Hydry oraz
   nowy notebook. Kryteria są obowiązkowe; wynik jednego nie kompensuje błędu
   innego. Ich realizację sprawdzają testy, a nie deklaracja wykonania zadania.
2. Przed zmianą zapisałem rzeczywisty przebieg referencyjny prostego pipeline'u:
   seed 7, limit MUTANG 4, próg ocen 3, wejście `FLYKWWIRIGRLKL`. Referencja jest
   teraz w `tests/mock_data/direct_reference.json`. Porównanie obejmuje dokładne
   sekwencje, kolejność, oceny i osobny batch końcowy.
3. Przeniosłem implementacje do `steps/<category>/strategies/`. Kategorie
   eksportują rzeczywiste klasy i funkcje przez jawne `__all__`. Usunąłem rejestry,
   managery, dekoratory rejestracji, stare deklaracje i puste klasy rodzin.
   Import strategii nie uruchamia modeli; usunąłem m.in. pośredni import TensorFlow
   z inicjalizatora BattleAMP.
4. Zastąpiłem konstrukcję przez nazwę algorytmu konstrukcją konkretnego obiektu.
   `Configurable` wspólnie obsługuje kopię konfiguracji i przygotowanie zasobów.
   Wartości domyślne wynikają z konstruktorów. Powielone dataclasses używane
   wyłącznie do przekazywania parametrów do fabryk zostały usunięte.
5. Rozdzieliłem odpowiedzialności: `PipelineSchema` przechowuje niemutowalne
   deklaracje i identyfikatory; `PipelineBuilder` dodaje, wstawia, podmienia,
   usuwa, zbiera zakresy, waliduje i buduje; `Pipeline` deleguje edycję oraz
   udostępnia ustawienia, opisy, grafy i wyniki. `build()` zwraca osobny
   `ExecutablePipeline`. Edycja deklaracji nie zmienia wcześniejszego buildu.
6. Przeniosłem kontrakty pól do komponentów. Walidator odróżnia pole kompletne od
   `OptionalField`, które może mieć brakujące wiersze. `optional_fields` oznacza
   zgodę na częściowe dane wśród pól wymaganych, a nie dodatkowy rejestr ani
   drugi opis parametrów. Sprawdza kolejność, gałęzie i nawrót pętli. Podsumowania
   przepływu eliminują ponowne przechodzenie poddrzew zagnieżdżonych pętli.
7. Zachowałem wspólny lifecycle `precompute` → `prepare_call` → `_execute` oraz
   centralne zliczanie i obserwacje w `Step.__call__`. Runner tworzy stan nowego
   wykonania i zarządza RNG, saverem oraz finalizacją. Progi ocen dotyczą całych
   partii: zakończona partia nie traci ocenionych peptydów po przekroczeniu limitu.
8. Przepiąłem zapis i odczyt na wspólny kontrakt SQLite/JSON. Ocenione archiwa,
   batch wychodzący, historia ocen, metadane komponentów i checkpointy są
   rozdzielone. Replay odtwarza stan i RNG oraz porównuje digest sekwencji.
   Usunąłem czytniki starych układów CSV/PT. Naprawiłem też aktualizację indeksów
   kroków trajektorii w wykonaniu sekwencyjnym.
9. Przepisałem konstrukcję orkiestratora na `_target_` Hydry. Presety są zwykłymi
   funkcjami; grupy YAML zawierają krótkie wywołania. Backend uruchomienia jest
   konkretną strategią, a szablon Slurm może pochodzić z serwera. Zamrożony tekst
   skryptu nie przechodzi ponownej interpolacji OmegaConf.
10. Ujednoliciłem prezentację `describe` / `describe_component`: `raw`, `pd`
    i domyślne `display`; opisy pochodzą z konstruktorów i dokumentacji Sphinx.
    Dodałem `docs/tutorial/wrapper_new.ipynb` oraz instrukcję. Istniejący
    `docs/tutorial/wrapper.ipynb` został zachowany bajt w bajt.
11. Przeniosłem testy naukowe do aktualnych ścieżek importu i zastąpiłem testy
    usuniętego API testami nowych kontraktów. Cofnąłem przypadkowe formatowanie
    niezwiązanych testów. Poprawiłem dwa skrypty pobierania APEX i deklaracje
    zasobów paczki, ponieważ wskazywałyby na usunięte katalogi.
12. Po testach sprawdziłem także wheel. Stary katalog `build/` wprowadzał do niego
    usunięte moduły, więc usunąłem wygenerowany katalog i zbudowałem paczkę
    ponownie. Końcowy wheel zawiera dokładnie aktualne źródła: porównałem bajty
    wszystkich 323 plików Python i sprawdziłem brak starych modułów. Z rozpakowanej
    paczki uruchomiłem rzeczywiste obliczenie referencyjne oraz kompozycję Hydry.

## Przegląd redundancji

Wspólna konstrukcja, kopiowanie konfiguracji, lifecycle, przygotowanie zasobów,
walidacja artefaktów, renderowanie i zapis mają pojedyncze implementacje.
`Lams`, `Tandem`, `Move`, `LpBebo` i `ESM` są funkcjami składającymi `Ranked`;
nie wymagają pustych podklas wybierających jedynie domyślne strategie.

Zachowane abstrakcyjne interfejsy definiują różne operacje matematyczne i ich
sygnatury: geometria, wybór kierunków, scoring, selekcja, kombinacja i dispatch.
Dataclasses opisują dane lub rzeczywiste podsumowania przepływu. Jedyną klasą
bez zachowania i pól w przeglądzie AST źródeł jest `DataUnavailableError`:
odróżnia brak zachowanych danych od błędu wykonania. Nie znalazłem identycznych
bajtowo modułów innych niż inicjalizatory pakietów.

Kod architektur wytrenowanych modeli został zachowany. Ich formaty checkpointów
mogą wymagać odczytu nazw klas zapisanych przez autorów modeli; nie są to aliasy
starego publicznego API PepCompass. Nie wykonywałem ogólnego przepisania tych
architektur ani ich skryptów treningowych.

## Weryfikacja wykonana na repozytorium docelowym

- Pełny `pytest`: **432 passed**, bez wyłączania modułów i bez pomijania testów.
- Ruff `F,E9,I` i kontrola formatowania: **196 plików** objętych refaktoryzacją.
- Notebook `wrapper_new.ipynb`: pełne wykonanie w kernelu Jupyter; obejmuje
  konstrukcję i edycję, Local Enumeration, LE-BO z APEX, SQLite oraz replay.
- Dokładna zgodność rzeczywistego przebiegu z referencją; porównanie predykcji
  czystego i oryginalnego APEX w istniejących testach.
- Złożenie 27 kombinacji operacji zgadza się z niezależnym modelem sekwencji.
- Złożenie przepływów pól: 4096 par podsumowań × 4 wejścia w modelu dwóch pól.
  Łączenie gałęzi sprawdzone dla pełnej i częściowej obecności pól.
- Test głęboko zagnieżdżonych pętli potwierdza jedną ocenę kontraktu na węzeł.
- CLI: rzeczywiste `plan` i `run`; testy lokalnego dispatchu, subprocessów,
  ponownego użycia wyników i generowania skryptu Slurm.
- Wheel: budowa, brak usuniętych modułów, zgodność źródeł oraz uruchomienie
  izolowane od checkoutu.
- `git diff --check` i porównanie hashy istniejących notebooków.

Podstawowe polecenie testów:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m pytest -q -o cache_dir=/tmp/pep-target-cache
```

Walidacja notebooka używała `jupyter nbconvert --execute` z wynikiem w `/tmp`;
oryginał notebooka nie był nadpisywany. Wheel powstał przez `uv build --wheel`.
Nie zmieniałem zależności projektu ani `uv.lock`.

## Zakres potwierdzenia i ograniczenia

Potwierdziłem nowy kontrakt i regresje CPU; nie jest to dowód poprawności wszystkich
algorytmów dla wszystkich danych. Nie uruchamiałem treningu, GPU, rzeczywistego
klastra Slurm ani wszystkich opcjonalnych modeli, w tym wymagających TensorFlow.
Nie ma skonfigurowanego typecheckera, więc go nie uruchamiałem. Nie wykonywałem
mutation testing. Szeroki lint importowanego kodu modeli nadal zgłasza istniejące
problemy; wynik Ruff powyżej dotyczy jawnie wskazanych 196 plików.

Walidator odwiedza każdy węzeł raz. Operacje na podsumowaniach zależą również od
liczby nazw pól: koszt wynosi O((N+E)F), a nie wykładniczo od głębokości pętli.
Tworzenie modeli i praca numeryczna mają osobny koszt. Edycja snapshotu kopiuje
strukturę deklaracji; nie inicjalizuje modeli.

Szczegółowe kryteria i dowody znajdują się w `../CONFORMANCE_REPORT.md`.

## Równoległa zmiana po testach

Po zakończeniu pełnego przebiegu testów pojawiła się niezależna zmiana w
`assets/tools/proteolysis_data_manager/src/proteolysis_data_manager/acquisition/half_life.py`:
backtick przed pierwszym docstringiem. Odczyt AST potwierdził `SyntaxError` w linii 1.
Nie zmieniałem tego pliku. Wynik 432 testów dotyczy stanu sprzed tej równoległej
edycji; ponowne uruchomienie całego repozytorium wymaga usunięcia tego błędu.
