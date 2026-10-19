# Audyt źródeł półtrwania i proponowany kontrakt pobierania

Historical pre-implementation audit (2026-10-01). Paths, proposed schemas and
filter counts below describe that audit, not the current pipeline. For the
implemented workflow, see [README.md](README.md). The audit receipt is now
packaged in the library's
[acquisition resources](../../../tools/proteolysis_data_manager/src/proteolysis_data_manager/acquisition/resources/audit_snapshot.json).

Stan: 2026-10-01. To raport przed implementacją downloaderów i przed decyzją
o integracji z `serum_proteolysis_calibration`. Istniejące skrypty, dane oraz
notebooki serum nie zostały zmienione. Publiczne pliki sprawdzone podczas audytu
pobrano do `/tmp/pep-half-time-audit`, poza docelowym katalogiem danych.
Checksumy i liczności sprawdzonych plików zapisano w `audit_snapshot.json`.

Zakres wynika z `tmp/Pepcompass - extend estimation problem (1).pdf`, w tym
hiperłączy osadzonych w PDF: PEPlife2, PepMSND, Cavaco i PlifePred/Mathur.
Cavaco opublikowano w 2021, Mathur w 2018; daty 2020 i 2017 z notatek odnoszą
się do wcześniejszych etapów publikacji.

## 1. Co już pobiera część serum

Trzy wejścia mają tę samą motywację: obserwowane półtrwanie całego peptydu,
używane do walidacji B. Nie są to pliki zawierające wyłącznie kolumnę półtrwania:
zawierają również sekwencje, modyfikacje i kontekst pomiaru.

| Klucz w rejestrze serum | Lokalny plik pod `data/serum_proteolysis_calibration/raw/` | Pobieranie | Liczba rekordów |
| --- | --- | --- | ---: |
| `peplife2_serum` | `peplife2/human_serum.json` | `download_peplife2.py human_serum` | 170 |
| `peplife2_plasma` | `peplife2/human_plasma.json` | `download_peplife2.py human_plasma` | 378 |
| `dramp_stability` | `dramp/stability_amps.xlsx` | `download_fixed_sources.py dramp_stability` | 110 |

Dowody w repozytorium:

- `../serum_proteolysis_calibration/config/serum_sources.yaml`: wszystkie trzy
  źródła są obecnie wymagane (`required: true`).
- `../serum_proteolysis_calibration/download_serum_sources.py`: uruchamia te
  źródła oraz pozostałe źródła kalibracji.
- `assets/experiments/analysis/09_26/20_serum_proteolysis_calibration_max/notebooks/0_data_analysis/0_12_serum_halflife_labels.ipynb`:
  czyta trzy powyższe ścieżki i definiuje lokalne parsowanie oraz filtry.
- `notebooks/_analysis_common.py` w tym eksperymencie: definiuje `SERUM_ROOT`,
  `SERUM_RAW`, `SERUM_PROCESSED`. Nie ma osobnego wspólnego loadera półtrwania.

Pozostałe źródła serum mają inne role: BRENDA/SABIO — kinetyka enzymów,
MEROPS/MMP-PICS — specyficzność cięcia, DPP4/qPISA — obserwacje proteolizy,
qMSP-MS/ADAMTS13 — pomiary ilościowe proteaz, HPA — obfitość. Nie należy
przenosić ich do zbioru etykiet półtrwania peptydów na podstawie samej obecności
danych czasowych.

### Liczności odtworzone z plików, nie tylko z zapisanych wyników notebooka

| Etap filtra PEPlife2 | Human serum | Human plasma |
| --- | ---: | ---: |
| Wszystkie zwrócone rekordy | 170 | 378 |
| Dokładna liczba + jednostka rozpoznawana przez notebook | 81 | 179 |
| Dodatkowo `in vitro` | 59 | 72 |
| Dodatkowo liniowy | 37 | 60 |
| Dodatkowo alfabet 20 standardowych aminokwasów | 32 | 48 |
| Unikalne sekwencje w ostatnim wierszu | 19 | 30 |
| Publikacje w ostatnim wierszu | 16 | 18 |

Filtr plasma odtworzono analogicznie do serum; notebook w swojej tabeli
coverage raportuje dla plasma wszystkie 378 rekordów, nie te 48.
Liczba 32 nie oznacza 32 niemodyfikowanych peptydów all-L: notebook pozostawia
modyfikacje, a jeden z tych rekordów ma chirality `Mix`.

DRAMP: 110 rekordów / 81 unikalnych sekwencji po normalizacji opisanej poniżej.
Filtry notebooka zwracają 64 rekordy / 49 unikalnych sekwencji / 11 publikacji.
Te 64 rekordy obejmują 53 liniowe i **11 cyklicznych**; policzone `is_linear`
nie jest użyte w masce `dramp_usable`. Gatunki w tych 64 rekordach:
27 human serum, 16 monkey serum, 12 fetal bovine serum, 8 mouse serum,
1 porcine serum. To nie jest zbiór 64 pomiarów ludzkiego serum.

Najważniejsze ograniczenia obecnego parsowania:

- Pierwsza liczba z jednostką w DRAMP nie musi być półtrwaniem. Np.
  `DRAMP29842`: „More than 70% of peptides remain intact after 6.5 h”
  zostaje potraktowane jak półtrwanie 6.5 h.
- `DRAMP29839`: półtrwanie „less than 0.5 h” staje się dokładnym 0.5 h;
  `about 1 h` i `∼1.5 h` tracą informację o przybliżeniu.
- PEPlife2: `<30`, `>24` itp. są odrzucane przez obecny parser liczb.
  W nowym pipeline powinny pozostać jako granice, szczególnie przy poszukiwaniu
  peptydów bardzo stabilnych.
- `is_modified` w PEPlife2 sprawdza `chem_mod`, ale nie końce `nter`, `cter`
  ani chirality. Sekwencja standardowych liter nie wystarcza do ustalenia chemii.
- Dedup po samej sekwencji jest zbyt mocny. W DRAMP `RRWQWR` ma różne
  modyfikacje końców i różne obserwacje stabilności.
- Brak nakładania 49 sekwencji DRAMP z 19 przefiltrowanymi sekwencjami serum
  nie oznacza niezależności źródeł. W całych źródłach są wspólne sekwencje
  i publikacje; odmienny parser może pozornie rozdzielić ten sam eksperyment.

## 2. Źródła z PDF i jak je pobierać

### PEPlife2 — cała baza jako warstwa źródłowa

[Dokumentacja API](https://webs.iiitd.edu.in/raghava/peplife2/usr_guide.html)
udostępnia filtry, nie opisuje endpointu `all`. Sprawdziłem dwa publiczne żądania:

```text
https://webs.iiitd.edu.in/raghava/peplife2/api/api.php?dataType=seq&dataValue=Natural
https://webs.iiitd.edu.in/raghava/peplife2/api/api.php?dataType=seq&dataValue=Modified
```

W bieżącym snapshotcie zwracają odpowiednio 2561 i 1939 rekordów. Zbiory ID są
rozłączne; suma to **4500 rekordów / 1808 unikalnych sekwencji**. Zbiór ID jest
dokładnie równy ID z publicznego
[pełnego FASTA](https://webs.iiitd.edu.in/raghava/peplife2/dataset/Peplife_complete.fasta).
Wszystkie 170 lokalnych rekordów serum i 378 plasma występuje w tych odpowiedziach,
z identyczną zawartością rekordów, nie tylko identycznymi ID.

Proponowany downloader zachowuje obie odpowiedzi bez zmian, łączy po source ID
i sprawdza kompletność wobec FASTA. FASTA służy do kontroli ID, bo nie zawiera
pełnych metadanych półtrwania. Gdy kontrola przestanie się zgadzać, download
nie powinien być oznaczony jako kompletny. Nie należy zakładać stałej liczby 4500.
Podziały serum/plasma/gatunek/in vivo mają być widokami utworzonymi po pobraniu.

### PepMSND — arkusz autorów, niezależnie od klasyfikatora

[Repozytorium autorów](https://github.com/hmenghu/PepMSND) zawiera
`Dataset/Dataset.xlsx`. Sprawdzony commit:
`c0fa6957c7f88a57af29410cfc075542c92a2c63`.

Arkusz: **635 rekordów / 558 unikalnych sekwencji**, 153 kolumny, w tym
`Sequence`, `SMILES`, modyfikacje, `Cyclic/Linear`, **`T1/2(hour)`**,
`Test Sample` i `In-Vivo/Vitro`. Wszystkie 635 wartości półtrwania są liczbowe;
brakuje osobnych kolumn PMID/DOI. To wartości udostępnione przez autorów —
bez publikacji źródłowej nie można dla każdej z nich potwierdzić, czy pierwotny
pomiar był dokładny, czy został uproszczony podczas kuracji.

Po ujednoliceniu pisowni jest 80 rekordów human serum (74 in vitro), 231 human
plasma; reszta obejmuje również szczury i małpy. 333 rekordy są in vivo, 303
in vitro. Tylko 373 sekwencje przechodzą prosty test alfabetu 20 aminokwasów.

Pobierać XLSX po przypiętym commicie, zachować metadane i SMILES, a cechy
modelowe pozostawić w surowym pliku. Nie pobierać wag ani środowiska modelu
do samego pozyskiwania etykiet. Snapshot ma także
[DOI Zenodo](https://doi.org/10.5281/zenodo.15687572), wskazany przez
[publikację](https://doi.org/10.1039/D5DD00118H).

### PlifePred / Mathur — dwa opublikowane arkusze

[Publikacja](https://pmc.ncbi.nlm.nih.gov/articles/PMC5983457/) udostępnia:

- [S1: naturalne i modyfikowane](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196829.s007&type=supplementary):
  **261 rekordów / 234 unikalne sekwencje**, półtrwanie w sekundach.
- [S2: naturalne](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0196829.s008&type=supplementary):
  **163 rekordy / 163 unikalne sekwencje**, półtrwanie w sekundach.

Weryfikacja arkuszy: wszystkie 163 sekwencje naturalne są obecne w zbiorze 261.
Nie sumować ich jako 424 niezależnych obserwacji. Naturalny ma 139 rekordów
liniowych i 24 cykliczne; mieszany ma 226 liniowych i 35 cyklicznych.
Modyfikacje, końce i chirality są w osobnych kolumnach. Nie ma osobnych kolumn
gatunku, matrycy ani PMID; nie oznaczać tych arkuszy automatycznie jako human serum.
Pole `id.pdb` zachować jako identyfikator autora; powiązanie z PEPlife2 wymaga
sprawdzenia zgodności pomiaru. Np. autorzy podają dla `1003.pdb` 1800 sekund,
a obecny PEPlife2 ID `1003` podaje `<30` minut. Nie są to równoważne etykiety.
Usunięcie HTML `<br>` w sekwencji musi być jawną, raportowaną normalizacją.

Arkusze są gotowymi benchmarkami po filtracji autorów, ale nie niezależnymi
zbiorami względem PEPlife. Zachować oba jako osobne widoki i raportować overlap.

### Cavaco — oddzielić meta-analizę, trening i walidację

[Publikacja](https://pmc.ncbi.nlm.nih.gov/articles/PMC8301568/) opisuje 579
wstępnych wpisów, **129 peptydów z 51 publikacji** do regresji, 16 peptydów
walidowanych eksperymentalnie oraz 21 peptydów terapeutycznych do drugiej
walidacji. To różne jednostki/liczności; 51 nie jest liczbą peptydów.

Publiczny PMC Open Data ma wersję `PMC8301568.1/`. Sprawdzono:

- [DOCX](https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/CTS-14-1349-s001.docx):
  23 tabele, głównie opis kategorii i analizy; nie prosty XLSX z 129 etykietami.
- [PDF suplementu](https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/CTS-14-1349-s003.pdf):
  zawiera zestawienie sekwencji, warunków, stabilności i produktów degradacji.
  Miesza półtrwania z procentem pozostającego peptydu po określonym czasie.
- [RAR](https://pmc-oa-opendata.s3.amazonaws.com/PMC8301568.1/CTS-14-1349-s002.rar):
  146216453 bajty, ok. 139.4 MiB. Nagłówki pokazują m.in. instalator
  `AminoAcids/Amino Acids Setup 1.0.5.exe` i pliki aplikacji Electron.
  Pełna lista wpisów RAR nie zawiera osobnego XLSX/CSV; zawiera `app.asar`,
  którego zawartości w tym audycie nie rozpakowywano.
  Nie należy utożsamiać rozmiaru tego pliku z rozmiarem datasetu.

Proponuję pobierać wybrane suplementy z PMC i wyodrębnić osobne widoki danych
z publikacji. PDF wymaga parsera tabel oraz sprawdzenia powiązania wiersza,
sekwencji, warunku i odczytu; nie należy automatycznie uznać każdej liczby z `h`
za półtrwanie. Jeśli nie uda się odzyskać wszystkich 129 rekordów w sposób
jednoznaczny, raportować częściowy zbiór i brakujące rekordy. Dla walidacji 16
peptydów zachować C-terminal amidation oraz protokół human serum. Nie szacować
brakujących półtrwań z wykresu bez osobnej decyzji.

## 3. Wstępne porównanie zbiorów

Poniższe porównanie dotyczy **sekwencji**, nie tożsamości cząsteczki ani
eksperymentu. Normalizacja: usunięcie znaczników HTML i białych znaków,
zamiana liter na wielkie. Nie rozwijano opisów niestandardowych reszt ani SMILES.
Odmienne konwencje zapisu mogą zmniejszać wykryty overlap. Wielkie litery same
nie dowodzą chirality L. Liczności dotyczą wszystkich wierszy źródeł.

| Zbiór | Rekordy | Unikalne sekwencje | Wspólne sekwencje z pełnym PEPlife2 | Sekwencje niewykryte w PEPlife2 |
| --- | ---: | ---: | ---: | ---: |
| PEPlife2 | 4500 | 1808 | 1808 | 0 |
| PlifePred natural | 163 | 163 | 163 | 0 |
| PlifePred natural+modified | 261 | 234 | 230 | 4 |
| PepMSND | 635 | 558 | 255 | 303 |
| DRAMP stability | 110 | 81 | 49 | 32 |

Pozostałe przecięcia: PepMSND × PlifePred natural = 134;
PepMSND × PlifePred mixed = 147; DRAMP × PepMSND = 23;
DRAMP × PlifePred natural = 20; DRAMP × PlifePred mixed = 32.
W szczególności 303 sekwencje niewykryte w PEPlife2 nie są automatycznie 303
nowymi niezależnymi eksperymentami. Cavaco nie policzono: brak ukończonej,
sprawdzonej ekstrakcji tabel pomiarowych.

Docelowy raport porównania powinien zawierać również:

- overlap tożsamości chemicznej: sekwencja + chirality + końce + modyfikacje
  + cyklizacja lub zweryfikowany SMILES;
- overlap eksperymentu: cząsteczka + publikacja + matryca/gatunek + warunki;
- zgodność etykiet po konwersji jednostek, osobno wartości dokładne i granice;
- liczbę wspólnych pomiarów, sprzecznych etykiet i pomiarów bez wystarczających
  metadanych do rozstrzygnięcia;
- rozkład długości/półtrwania i lejek filtrów dla każdej wersji datasetu;
- ochronę przed przeciekiem: podział po publikacji i grupach podobnych peptydów.

## 4. Kontrakty znalezione w bibliotece i proponowany format

W `src/pep_compass` nie znalazłem istniejącego schematu datasetu półtrwania ani
loadera PEPlife/DRAMP. `data/dataset.py: ExperimentDataset/TableHandle` opisuje
wyniki eksperymentów optymalizacyjnych; nie jest kontraktem wejściowego zbioru
pomiarów. `optimization/stability_estimation/estimation.py` estymuje pamięć
batchy, mimo nazwy nie estymuje biologicznego półtrwania.

Istotne istniejące kontrakty:

- `optimization/components/helpers/proteolysis/cleavage.py`:
  `MEROPS_AMINO_ACIDS` — alfabet standardowych reszt. Zachować oryginalny zapis,
  a zgodność z alfabetem zapisać jako flagę.
- `optimization/components/helpers/proteolysis/dynamics.py`:
  `ProteaseKinetics`, intensywności zdarzeń, `first_cut_distribution`.
  `1 / total_hazard` to średni czas pierwszego cięcia. Półtrwanie nietkniętego
  peptydu przy stałym hazardzie wynosi `ln(2) / total_hazard`; tych etykiet nie
  należy zamieniać. Obserwacje in vivo mogą opisywać eliminację, nie samo cięcie.
- `utils/sequence_utils.py: to_one_hot`: przycina sekwencje do 25 reszt.
  Nie używać tej funkcji do normalizacji źródeł; zachować pełne długości.
- `../serum_proteolysis_calibration/_common.py`: atomowy zapis raw i manifest
  z SHA-256, URL, rozmiarem i czasem pobrania. Zachować tę konwencję, ale nie
  importować ścieżki `acquisition_root()` serum do nowego pipeline.

Proponowany kontrakt: **jeden wiersz = jedna obserwacja w określonych warunkach**.
Nie jeden wiersz na unikalną sekwencję. Surowe pliki są zachowywane bez zmian.
Tabela znormalizowana TSV/Parquet ma wersję schematu i następujące pola:

| Grupa | Pola |
| --- | --- |
| Pochodzenie | `observation_id`, `source`, `source_version`, `source_record_id`, `source_file_sha256`, `source_locator`, `publication_id`, `dataset_view` |
| Peptyd | `sequence_raw`, `sequence`, `length`, `chirality`, `topology`, `n_terminal_modification`, `c_terminal_modification`, `chemical_modifications`, `smiles`, `is_standard_sequence` |
| Warunki | `species`, `matrix`, `in_vivo_in_vitro`, `assay`, `temperature_c`, `ph`, `peptide_concentration_raw`, `matrix_fraction_raw`, `conditions_raw` |
| Pomiar | `measurement_kind`, `half_life_raw`, `unit_raw`, `relation`, `half_life_seconds`, `lower_bound_seconds`, `upper_bound_seconds`, `remaining_fraction`, `observation_time_seconds` |
| Kontrola | `parse_status`, `exclusion_reasons`, `normalization_notes`, `chemical_identity_status`, `duplicate_group_id` |

`measurement_kind` odróżnia półtrwanie od pomiaru pozostającej frakcji.
`relation` odróżnia `eq`, `lt`, `le`, `gt`, `ge`, `approx`, `interval`, `unknown`.
Braki metadanych pozostają brakami. Nie dopowiadać human serum na podstawie
tematu artykułu. `k_obs_per_second = ln(2)/t_half_seconds` wyliczać tylko dla
dodatnich, dokładnych etykiet w widoku przyjmującym model pierwszego rzędu.
Peptydy modyfikowane i cykliczne pozostają w bazie; widoki decydują o użyciu.

Proponowana struktura, jeszcze nieutworzona:

```text
data/half_time_datasets/
  raw/{peplife2,pepmsnd,plifepred,cavaco,dramp}/<snapshot>/
  manifests/
  processed/observations.parquet
  processed/observations.tsv
  views/                         # oryginalne benchmarki i jawne filtry
  reports/{coverage,overlap,label_conflicts,excluded_rows}/
```

Skrypty do przygotowania po decyzji integracyjnej: `download_peplife2.py`,
`download_paper_datasets.py`, `download_dramp.py`, `normalize_datasets.py`,
`compare_datasets.py` oraz wspólny orkiestrator. Konfiguracja źródeł powinna
określać URL/commit/wersję, oczekiwane kolumny i profile filtrów. Pobieranie nie
wymaga uruchomienia notebooków ani budowy manifestu proteaz serum. Błędy źródła,
niepełny download i niezgodność schematu mają kończyć się niezerowym statusem.
Ponowne użycie snapshotu zachowuje pierwotny czas pobrania; czas weryfikacji
powinien być osobnym polem.

## 5. Decyzja dotycząca serum — warianty bez kolizji

Rekomendacja: docelowo `half_time_data` jest właścicielem pobierania powyższych
trzech źródeł, a serum korzysta z jawnego widoku danych półtrwania. Przejście
warto wykonać w dwóch krokach:

1. Dodać nowy pipeline bez ruszania istniejących źródeł. Import lokalnego
   snapshotu PEPlife2/DRAMP powinien zachować checksum i oryginalny manifest;
   nie oznaczać lokalnego kopiowania jako świeżego pobrania z sieci.
2. Dopiero po sprawdzeniu wyników przełączyć serum na nowego właściciela.
   Zachować legacy JSON/XLSX jako eksport zgodności lub jawnie zmienić
   `0_12` na czytanie nowej tabeli/widoku. Starsze wyniki oznaczyć jako wyniki
   poprzednich filtrów; poprawki DRAMP celowo zmienią liczności.

Alternatywa: pozostawić dwa niezależne pipeline i osobne katalogi. Zachowuje
obecne notebooki, lecz wymaga kontroli wersji/checksum, ponieważ snapshoty mogą
się rozjeżdżać. Nie wystarczy zmienić w YAML `acquisition_root`: obecny
orkiestrator nie przekazuje tego pola do downloaderów.

Jeśli zdecydujemy o centralizacji, lista zmian w serum obejmuje konkretnie:

- routing `peplife2_serum`, `peplife2_plasma`, `dramp_stability` w orkiestratorze
  oraz rejestrze; pozostałe źródła zachowują swoje role;
- ścieżki/loader w `0_12_serum_halflife_labels.ipynb` i ewentualne nowe stałe
  w `_analysis_common.py`, bez przepięcia całego `SERUM_RAW`;
- zachowanie dostępności starych plików dla wcześniejszych analiz;
- testy trzech tras w `tests/assets/scripts/downloads/serum_proteolysis_calibration/test_downloaders.py`,
  README serum i opisy provenance w notebooku;
- brak podwójnego liczenia: serum/plasma jako widoki pełnego PEPlife2, naturalny
  PlifePred jako podzbiór mieszanego, cross-source overlap jako powiązania.

W tym etapie nie przeniesiono ani nie usunięto żadnego datasetu. Implementacja
i przepięcie serum pozostają kolejnym etapem po wyborze powyższej strategii.
