# Weryfikacja scorerów geometrii latentnej

Ten katalog jest jedynym miejscem implementacji scoringu TANDEM, LAMS i MOVE.
Klasy zwracają wyłącznie score przypisany każdemu kandydatowi. Threshold,
nucleus i inne reguły wyboru znajdują się w `ranked/selection`.

## TANDEM

Do sprawdzenia przez autorów implementacji:

- czy reprezentacja kierunku ma pozostać `onehot`, czy powinna być różnicą
  `target - parent`;
- czy projekcja ma używać pełnej obciętej pseudoodwrotności Jacobianu;
- czy transformacje par `taken/taken` i `taken/not-taken` odpowiadają dokładnie
  eksperymentom z pracy magisterskiej;
- czy średnia ma obejmować wyłącznie pary zawierające co najmniej jedną mutację;
- czy pojedyncza mutacja powinna otrzymywać score równy zero.

## LAMS

Do sprawdzenia:

- czy score kombinacji jest minimum podobieństw wszystkich par mutacji;
- czy kombinacja z jedną mutacją powinna otrzymywać `+inf`;
- czy domyślny próg `0.15` pochodzi z właściwego wariantu eksperymentu;
- czy LAMS ma korzystać z tej samej reprezentacji projekcji co TANDEM.

## MOVE

Do sprawdzenia:

- założenie pierwszego rzędu, że przesunięcia pojedynczych mutacji można sumować;
- użycie ujemnej normy jako score, czyli większy score oznacza mniejszy ruch;
- czy wszystkie single mutations powinny być kodowane jednym batchem;
- czy odległość ma pozostać euklidesowa w przestrzeni latentnej.

## Wspólne ograniczenia

Redukcja produktu do `maximum_candidates` odbywa się przed materializacją.
Obecny baseline usuwa losowe alternatywy z największej grupy. Należy ustalić,
czy reprodukcja wyników wymaga zachowania tej losowości, czy deterministycznego
ograniczania według score pojedynczych mutacji.
