# Raport z agregowania danych do estymacji half life z peptydów

## Wstęp

Analize przeprowadziłem w następujacy sposób:
- dla każdego datasetu
  - przejrzenei typów danych, pobraenei odpowiednich datasetów, przejrzenie api które to umozliwiają 
  - następnie zrozumeinie jaei dane majmy do dyspozycji **ustalenie wspólnego formatu danych**, który umożliwa analize (na ten moment per typ danych)
  - **ustalenie sposobu na agregacje danych z różnych źródeł** oraz analiza:
    - kompletnośc (ile czego jest) 
    - zgodności (wspólne sekwencje, czxy jest rozjazd, czasami mam jakiś *baseline* w postaci dokładniejszego artykułu) 
    - końcowego audtydu
  - zrobienie końcowego dataseta który eksportuje 
- potem na wspólnych dataseteach
  - zrobienie odpowiednich relacji pomiedzy datasetami
    - windows vs kinetics 
    - halftime vs odpoweidnie klasy 
    - uwspolnienie nazw pomiedzy datasetami (w senei jak są wspóle poal to trzeba zadeklrawować)

## Indywidualne analizy

### Half-time

#### Dane 
Tutaj korzystam głownei z peplife2.0, oraz z dodatowych artkyułow, które zawierają trochę roszerzone datasety, przefiltrowane do ternowania niektórch modeli
- peplife2.0
- dramp
- pepmsnd
- **cavaco**
  - jest to niezależny **dataset**, ktory zaweira dokładny pomia

#### Pobranie 
**peplife2.0**
- zawiera oficjalne [api](https://webs.iiitd.edu.in/raghava/peplife2/api/rest.html?utm_source=chatgpt.com), więc można bezpośrednio wszystko pobrać
- dwuznazcne dane były zweryfikowane za pomocą chata z artykułami (#TODO - odnośnik gdzie to jest opisane) 

**pozostałe**
- w artykułach były dostarczone informacje związane z wyjściowymi danymi 

#### Analiza 
- najlepszy dataset to jets cavaco - mało peptydów **ale z rozkładami**
-  dane są spójne (nie ma rozjazdu w opisie) 
-  widzimy że możemy wyprowdziać wzór który **estymuje nam przedział ufności danej obserwacji**. Pomysł tego poelgal an tym, że możęmy wtedy zamiast MSE i błędów liczbowyhc (które moim zdaniem sa torhce słąbym estymatorem w tym przypadku) mozęmy srpwaiczc doadtkwo **czy trafiamy w określony przedział**. 


