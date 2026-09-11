# Weryfikacja scorerów modelowych

## Decoder likelihood / LPBEBO

Implementacja przypisuje kombinacji sumę logarytmów prawdopodobieństw residuów
z dekodera warunkowanego latentem sekwencji rodzica. Selekcja nucleus jest
osobną strategią i nie należy do scorera.

Do sprawdzenia przez autorów implementacji:

- zgodność indeksów alfabetu, paddingu i długości maksymalnej z HydrAMP;
- czy wynik powinien być sumą, czy średnią po zmienionych pozycjach;
- czy residua niezmienione powinny wnosić wkład do score;
- czy dekoder powinien być warunkowany wyłącznie rodzicem;
- wartości domyślne `top_p=0.9` i `temperature=1.0` dla presetu LPBEBO.

## ESM plausibility

Do sprawdzenia:

- czy stosowana wartość jest rzeczywistym pseudo-log-likelihood, ponieważ
  obecny kod wykonuje pojedynczy forward bez maskowania kolejnych pozycji;
- właściwy model ESM2 i wersja `fair-esm`;
- czy agregacja po pozycjach ma pozostać średnią;
- wartości progów używane w eksperymentach oraz urządzenie wykonania.
