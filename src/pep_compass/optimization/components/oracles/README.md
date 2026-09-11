# Oracles — zakres zmian do przekazania

## Wstęp 

Poprawnie jest zaimplementowany, APEX (`src/pep_compass/optimization/components/oracles/strategies/apex`)  


## To co u mnie LLM sugeruje - nie sprwadzłęm kontekstu 

Instrukcje dla osoby przejmującej porządkowanie tego komponentu, bez konieczności samodzielnego
czytania każdej metody w `strategies/`.

### Scope

Ten dokument opisuje wyłącznie strukturę i spójność kodu adapterów (`oracles/strategies/*/oracle.py`),
nie poprawność naukową predykcji poszczególnych modeli (wagi, preprocessing, interpretacja wyniku).
Stan walidacji naukowej opisuje [`docs/developer/REFACTORING_HANDOFF_PL.md`](../../../../../docs/developer/REFACTORING_HANDOFF_PL.md),
sekcja "Modele predykcyjne oracle" — modele czekają na osobną walidację przez ich właścicieli.

### Warstwa wspólna — nie zmieniać

`Oracle` (`base.py`) → `BlackBoxOracle` (`strategies/black_box.py`) → rejestracja przez
`_black_box_oracle()` w `strategies/__init__.py`. Ta warstwa jest spójna dla wszystkich sześciu
zarejestrowanych oracle'i (`apex`, `battleamp`, `eipred`, `hydrophobicity`, `mbc_attention`,
`toxipep`): limit wywołań (`context.state.remaining_oracle_calls()`), chunkowanie po
`evaluation_batch_size`, zapis obserwacji (`record_observations`), konwencja pól
`oracle.<name>.score` / `.direction` / `.name`. Żaden pojedynczy oracle tego nie duplikuje. Każdy
`OracleManager.register(...)` występuje dokładnie raz — brak duplikatu rejestracji, w
przeciwieństwie do analogicznych bugów opisanych dla `walkers`/`mutation_generators` w planie
restrukturyzacji tamtych komponentów.

### APEX — model wag wprowadzony jako nazwany wariant

`APEXBlackBox`/`PredictorAPEX` wcześniej wymagały ręcznie skopiowanych folderów wag
(`APEX_pathogen_models`, `Full_APEX_pathogen_models`) bezpośrednio w pakiecie `apex/` — oba były puste
(tylko `.gitignore`, bez wag), więc oracle był niesprawny przy każdym świeżym checkoucie.

Wczytywanie wag zostało przepisane analogicznie do `AutoencoderFactory`/`AutoencoderRegistry` w
HydrAMP: `PredictorAPEX(model=...)` wybiera nazwany wariant z `APEX_MODEL_VARIANTS`
(`apex/APEX_predictor.py`), każdy wariant wskazuje na `apex/models/<nazwa>/` (analogicznie do
`autoencoder/strategies/hydramp/models/<nazwa>/`). Dostępne warianty:

- `default` — 8-patogenowy zestaw z oryginalnego artykułu APEX (pliki `APEX_*`), źródło:
  [`Yimeng-Zeng/APEXGo`](https://github.com/Yimeng-Zeng/APEXGo). Instalacja:
  `assets/scripts/downloads/apex/download_apex_models_default.sh`.
- `full` — rozszerzony zestaw 34 patogenów, 40 modeli (pliki `trained_all_model_*_ensemble_*`), źródło:
  oficjalne, kanoniczne repozytorium
  [`machine-biology-group-public/apex`](https://gitlab.com/machine-biology-group-public/apex/-/tree/main/trained_models)
  (~27 MB/plik, ~1 GB łącznie). Instalacja:
  `assets/scripts/downloads/apex/download_apex_models_full.sh`.

Oba skrypty pomijają pobieranie (exit 0), jeśli pliki wag już są obecne w katalogu docelowym — nie
pobierają ponownie przy kolejnych uruchomieniach. Każdy katalog wariantu ma własny `.gitignore`
(`*` poza `.gitignore`/`README.md`) — wagi nigdy nie trafiają do repozytorium, niezależnie od
katalogów `.gitignore` wyższego poziomu.

Config (`oracle: {method: apex, parameters: {model: default, ...}}`) musi teraz podawać `model`
jawnie — `strategies/__init__.py`'s `build_apex` przyjmuje ten parametr
(`_COMMON | {"mic_aggregate", "mic_bacteria", "model", "device"}`).

Usunięto martwą klasę `HydrAMPAPEXBlackBox` (duplikowała `APEXBlackBox`, nigdzie nieużywana) —
wcześniejszy punkt 4 tej listy jest już zamknięty.

**Nieprzetestowane uruchomieniowo w ramach tego przeglądu**: rzeczywiste pobranie i wczytanie wag
(skrypty nie zostały odpalone — plik `full` to ~1 GB, decyzja o pobraniu należy do osoby uruchamiającej
eksperyment) oraz poprawność liczbowa predykcji.

### Zmiany do wprowadzenia w pozostałych pięciu oracle'ach

#### 1. Brakujące `self.maximize`

`BlackBoxOracle._attach_result` czyta `getattr(self.black_box, "maximize", False)` — brak
przypisania w `__init__` oznacza cichy fallback na `"minimize"`, bez jawnej decyzji w kodzie.

Dotyczy `strategies/eipred/oracle.py` (`EIPredBlackBox.__init__`) i
`strategies/mbc_attention/oracle.py` (`MBCAttentionBlackBox.__init__`) — obie klasy nie ustawiają
`self.maximize`. Oba modele dziś zwracają `log2(...)`, więc kierunek jest prawdopodobnie `False`, ale
wymaga potwierdzenia merytorycznego z właścicielem modelu przed dodaniem jawnego przypisania.

#### 2. Sprzeczny opis kierunku w `ToxiPepBlackBox`

`strategies/toxipep/oracle.py:27-32` — docstring klasy opisuje wyższy score jako bezpieczniejszy
peptyd (sugeruje maksymalizację). Komentarz przy `self.peptide_scorer` (linia 67) mówi
"for minimization", a `self.maximize = False` jest ustawione na sztywno (linia 73). Do wyjaśnienia
merytorycznie z właścicielem modelu, który z dwóch opisów jest poprawny, i poprawienia drugiego.

#### 3. Nieużywany `self.cache`

Występuje w `apex/oracle.py`, `battleamp/oracle.py`, `eipred/oracle.py`, `hydrophobicity/oracle.py`,
`mbc_attention/oracle.py`, `toxipep/oracle.py` (ten ostatni ma dodatkowo gettery
`get_cache_size`/`get_cached_results`). Nic poza samym plikiem oracle'a nigdy go nie czyta; rośnie bez
ograniczeń przez cały przebieg optymalizacji. Do usunięcia z wszystkich sześciu plików, albo — jeśli
ma wartość diagnostyczną — do scentralizowania w jednym miejscu zamiast kopiowania w każdym adapterze.

#### 4. Ręczne smoke-testy w plikach produkcyjnych

`strategies/hydrophobicity/oracle.py` (linie 65-111) i `strategies/toxipep/oracle.py` (linie 132-182)
zawierają bloki `if __name__ == "__main__":` z ręcznymi testami i `print()`. Do usunięcia albo
przeniesienia do `tests/` jako właściwe testy jednostkowe.

#### 5. Niespójne źródło `AbstractBlackBox`

`apex/oracle.py` i `battleamp/oracle.py` importują `AbstractBlackBox` z `poli.core.abstract_black_box`;
`eipred/oracle.py`, `hydrophobicity/oracle.py`, `mbc_attention/oracle.py`, `toxipep/oracle.py`
importują z `poli_baselines.core.abstract_solver`. Do wyjaśnienia: czy rozjazd jest zamierzony (dwie
różne biblioteki dla różnych klas modeli), czy przypadkowy i wymaga ujednolicenia.

#### 6. Ręczna izolacja procesowa tylko w BattleAMP

`battleamp/oracle.py:47-52` uruchamia predyktor w osobnym procesie CPU przy `device=cuda` przez
ręczny `ProcessPoolExecutor` (komentarz w kodzie: unika konfliktu CUDA/cuSOLVER z PyTorchem
wykonującym LE-BO na GPU). Wszystkie sześć oracle'i przyjmują już `force_isolation` jako parametr POLI
(`_COMMON` w `strategies/__init__.py:10-13`). Do wyjaśnienia: czy wbudowany mechanizm POLI faktycznie
nie wystarcza tutaj (i wtedy ten wzorzec wymaga udokumentowania i ewentualnego powtórzenia tam, gdzie
jest potrzebny), czy to obejście możliwe do zastąpienia samym `force_isolation=True`.

### Powiązane dokumenty

- [`docs/developer/REFACTORING_HANDOFF_PL.md`](../../../../../docs/developer/REFACTORING_HANDOFF_PL.md)
  — stan walidacji naukowej modeli i kontekst poprzedniej restrukturyzacji pakietu.
