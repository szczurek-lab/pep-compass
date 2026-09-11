# Walidacja strategii oracle

## Wstęp

Ten katalog docelowo zawiera adaptery i duże implementacje modeli używanych jako
oracle. Podczas migracji stare pliki pozostają źródłem odniesienia, aby można
było porównać wyniki przed usunięciem starej struktury.

Przejrzałem jedynie APEX'a, żeby był przykąłd jak to trzeba zrobić 

## Struktura
- plik z opisem BlackBoxa (`oracle.py`)
- plik z działaniem modelu (`predictor.py`) 
- plik który zarzadza modelami (`model_registry`)
- folder z modelami (nazwa folderu - nazwa wywołania) 
  - KONIECZNIE dodawać skrypty do pobierani tego (`assets/scripts/downloads`), inaczej bardzo cięzko się weryfikuje kto co używał ostatecznie (on i szukanie tego ...) 

# Chat work - tego nie weryfikowałem
Nie czytałem jak te model tylko przeniosłęm (chyba) w odpowiednie miejsce, jeżeli na tym byśmy chcieli pracowac to trzeba todo struktury dopasować

## Zweryfikować

### APEX

Predyktor aktywności przeciwdrobnoustrojowej agregujący przewidywane wartości
MIC dla wybranych modeli patogenów.

Kod: [`apex/oracle.py`](apex/oracle.py), [`apex/`](apex/).

#### Co sprawdzić

- sposób agregacji `mean` i `max`;
- znaczenie oraz kierunek optymalizowanej wartości;
- zgodność indeksów modeli bakterii z oryginalną implementacją;
- preprocessing, długość sekwencji i obsługiwany alfabet;
- zgodność batchowego wyniku z wywołaniem pojedynczych sekwencji;
- urządzenie, typy tensorów i sposób ładowania wag.

#### Walidacja

Osoba walidująca powinna opisać źródło modelu, użyte wagi, wykonane porównania,
znalezione różnice i ich wpływ na użytkownika.

### BattleAMP

Model przewidujący aktywność przeciwdrobnoustrojową peptydu.

Kod: [`battleamp/oracle.py`](battleamp/oracle.py), [`battleamp/`](battleamp/).

#### Co sprawdzić

- zgodność preprocessingu z kodem źródłowym BattleAMP;
- interpretację logarytmu i kierunek optymalizacji;
- ograniczenia długości oraz alfabetu;
- zgodność wersji TensorFlow/Keras i zapisanych wag;
- zgodność ewaluacji batchowej i pojedynczej.

#### Walidacja

Należy wyjaśnić użytkownikowi znaczenie wyniku, wymagania środowiska oraz każdą
różnicę względem publikowanej implementacji.

### ToxiPep

Model oceniający toksyczność sekwencji peptydowej.

Kod: [`toxipep/oracle.py`](toxipep/oracle.py), [`toxipep/`](toxipep/).

#### Co sprawdzić

- preprocessing grafu i cech atomowych;
- wybór pliku wag oraz progu klasyfikacyjnego;
- znaczenie zwracanego score'a;
- obsługę błędnych i nietypowych sekwencji;
- zgodność CPU/GPU i ewaluacji batchowej.

#### Walidacja

Należy opisać, czy wynik jest prawdopodobieństwem, logitem czy etykietą, oraz
jak powinien być interpretowany w konfiguracji filtrów i oracle.

### Hydrophobicity

Deterministyczna funkcja obliczająca hydrofobowość peptydu na wybranej skali.

Kod: [`hydrophobicity/oracle.py`](hydrophobicity/oracle.py),
[`hydrophobicity/`](hydrophobicity/).

#### Co sprawdzić

- definicję każdej dostępnej skali;
- zachowanie dla nieobsługiwanych aminokwasów;
- normalizację względem długości sekwencji;
- kierunek optymalizacji.

#### Walidacja

Należy podać wzór lub źródło skali i wyjaśnić jednostkę zwracanego wyniku.

### EIPred i MBC-Attention

Dodatkowe predyktory biologiczne obecne w starym kodzie, ale jeszcze niewłączone
do nowego buildera.

Kod: [`eipred/oracle.py`](eipred/oracle.py),
[`mbc_attention/oracle.py`](mbc_attention/oracle.py).

#### Co sprawdzić

- pochodzenie i licencje modeli oraz danych pomocniczych;
- kompletność wymaganych artefaktów;
- zgodność importu `AbstractBlackBox` z aktualnym POLI;
- preprocessing i interpretację wyjścia;
- możliwość deterministycznego testu na małym fixture.

#### Walidacja

Po walidacji należy zdecydować, czy komponent jest oracle, filtrem biologicznym,
czy strategią `decision_models`; samo położenie w starym `models/` nie rozstrzyga
jego roli w nowej architekturze.
