# Weryfikacja polityk optymalizacyjnych

## ROBOT

Kod został przeniesiony bez zmiany metody numerycznej. Przed uznaniem go za
implementację referencyjną należy sprawdzić:

- zgodność wariantu MAP4 i parametrów fingerprintu z eksperymentami;
- użycie `LogExpectedImprovement` oraz orientację `maximize`;
- sens `best_f=train_y.min()` dla minimalizowanych funkcji celu;
- wpływ opcjonalnej standaryzacji wyniku GP;
- próg różnorodności Levenshteina i zachowanie przy batchu większym niż jeden;
- zgodność parametrów inicjalizacji losowej z artykułem;
- zachowanie kernela Tanimoto dla batcha o rozmiarze jeden.

## Trust region

Do sprawdzenia:

- czy promień ma być stały, jak w konfiguracji referencyjnej `origin/dev`, czy
  dynamicznie aktualizowany;
- czy sukces i porażka są liczone po każdym kandydacie, czy po całym batchu;
- orientacja funkcji celu przy aktualizacji najlepszego punktu;
- osobne znaczenie promienia dla odległości sekwencyjnej i latentnej;
- czy wyczerpanie minimalnego promienia powinno kończyć eksperyment.

## Fingerprinty i kernel

Pliki w `helpers` pochodzą ze starszej implementacji ROBOT. Zawierają kilka
wariantów fingerprintu, mimo że aktualnie używany jest tylko `Map4Fingerprint`.
Po walidacji należy usunąć nieużywane warianty i ujednolicić ich docstringi.
