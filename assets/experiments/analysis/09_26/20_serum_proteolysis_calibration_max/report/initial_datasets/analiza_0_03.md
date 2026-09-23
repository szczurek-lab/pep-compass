# Analiza 0_03

## Ile proteaz można znaleźć w BRENDA (czy istnieje indeks EC) oraz ich jakość

Z 54, które są hydrolazami, mamy 42 z indeksem EC, a 12 nie da się wyszukać w rekordach BRENDA.

Około $22.3%$ danych z BRENDA możemy przypisać do danych MEROPS.

### Komentarze

* **W jaki sposób deklarowane są labele:** rekord BRENDA zawiera typ pomiaru (`kcat`, `Km` albo `kcat/Km`), jego wartość oraz opis substratu. Żeby rekord mógł być labelem dla MEROPS, trzeba dodatkowo jednoznacznie odtworzyć sekwencję substratu oraz miejsce cięcia. Dopiero wtedy można zbudować odpowiadające mu okno MEROPS i utworzyć parę `MEROPS score → zmierzona wartość kinetyczna`. Typ pomiaru pozostaje jawnie zapisany jako `measurement_type`; nie traktujemy `Km`, `kcat` i `kcat/Km` jako tego samego labelu.

## Pokrycie BRENDA dla poszczególnych peptydaz

Tutaj wyniki są **tragiczne**, widzimy, że większość sekwencji jest bardzo słabo pokryta.

### Komentarze

* **W jaki sposób to się zlicza:** dla każdej peptydazy zliczamy nie surową liczbę rekordów BRENDA, lecz liczbę **unikalnych substratów z jednoznacznie odtworzonym miejscem cięcia**, czyli takich, dla których można zbudować konkretne okno MEROPS. Osobno można zliczać wszystkie rekordy z rozpoznaną sekwencją, ale bez miejsca cięcia nie są one bezpośrednimi przykładami do regresji `MEROPS → kinetyka`. Audyt ma właśnie raportować liczbę obserwacji, unikalnych sekwencji i unikalnych kontekstów miejsca cięcia dla każdej proteazy.

## Decyzje

Mało peptydaz możemy **porządnie** zweryfikować tą metodą. Trzeba się dopytać Pauliny, czy badanie tylko tych specyficznych peptydaz jest sensowne.

Brałbym peptydazy powyżej tych 4/5 (4, żeby zwiększyć liczbę peptydaz), żeby to miało jakikolwiek sens.

### Inna konstrukcja na tych danych

Możemy zaobserwować inne wnioski na podstawie tych danych, ponieważ ich ilość relatywnie do MEROPS-a też jest dosyć spora.
