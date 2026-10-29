# Biblioteka 

## Ogólnie 

### Co zmieniono
- poprawiona struktura grafu 
- ujednolicenie importów poprzez moduł steps, oraz ustalenie formatu zaproponowanego (całe szczęście, że to się stało) przez Adama
- notebooki tutorialowe
  - wrapper przejrzałem cały, opisuej cała abstrakcje - kazdy to powinien przeczytać ze zrozumieniem
  - są jeszzce tutoraiel od readera i configów, ale jes tylko na szybko przejrzałem - ktoś (Adam) powinien tą funkcjonalność zwalidować
- dodałem ujednolicenie danych, to jest:
  - są teraz zrobione dane związane z 


## Co trzeba przejrzeć: 

### [Orchestrator](https://github.com/maxi7524/pep-compass/tree/dev_max_integrated/src/pep_compass/orchestrator) (zarzadznie runnami z yamla)

Ogólnie wizje miałem taką:
- mamy ogólnie zdefiniowy moduł który **przyjmuje rózne konfiguracje uruchomienia** (wtedy możemy miec backend_slurm_<nazwa_serwera>_<wariant>) i możemy przy runnie podpinać te konfiguracje i jest to w bardzo proste - raz walidujemy skrypt


Nie miałem czasu tego przejrzeć i przestestować - zostawia to tobie Adam :) - to jest priorytet do zrobienia, żeby **ukończyć bazowy kod biblioteki**. 

### [predefined](https://github.com/maxi7524/pep-compass/tree/dev_max_integrated/src/pep_compass/predefined) (konfiguracja podstawowych algorymtów)

Tutaj musisz przejrzeć czy dobrze jest zadefiniowane lebo - warto się upewnić że farmazonwu nie ma 

### [Format sqltie](https://github.com/maxi7524/pep-compass/blob/dev_max_integrated/src/pep_compass/optimization/saving/backends.py) (WAŻNE - chce żebyś potwierdził ten format sqlite'a)

Tutaj do potwierdzeina tudzież ustalenia są dwie rzeczy:
- format danych końćowych
- jakie informacje śledzimy w trakcie runnu

Trzeba uzgodnic czy to jest dobra wersja wyników. Jest to ważne, poniewaz w przpyadku robienia kolejnych analiz oraz następnie udostępniania ich w publicznych zasobach, **będziemy musieli zachować kompatybilność wsteczna dla różnych wersji**. Formalnie nie jest to skomplikowane, (jest specjalny folder [schemas](https://github.com/maxi7524/pep-compass/blob/dev_max_integrated/src/pep_compass/optimization/saving/schema.py), który odpiaad za to żebyweidziec jaki foramt czytać), ale fajnie nie robić od razu śmietnika. 

Abstrakcje masz opisana w `docs/tutorial/reader_tutorial.ipynb` - nie bawiłem się tym dużo, sprawdziłem jedynie na szybko czy wszystok się poprawnie wykonuje. 


### [wszystkie zaimplementowane kroki (steps)](https://github.com/maxi7524/pep-compass/tree/dev_max_integrated/src/pep_compass/steps) 

Ogólnie tam jest bałagan na pewno ponieważ:
- (lipicec / sierpień) najpierw przenosiłem z 3 repo implmentacje żeby to ujednolicic 
  - robiłem to na w taki sposób, że kazałem przepisywą te funkcje prawie jeden do jeden, żebym móc zweryfikować, czy kod sie zgadza
  - następnei dopiero myślałęm jak rodzielic abstrakcje tych komponentów 
  - nazewnictwo jest niespójne, poniewaz wtedy był u mnei trochę panika / chaos, więc nie rodzieliłem tego idealnie - nazwy mogą być niespójne
  - #TODO - trzeba zatem
    - ujednolicić nazewnictwo (i dokumentacje z tymi stepami zrobić sensowną) 
    - sprwadzi funkconanośc poszczególnych komponentów


### [Dokumentacja](https://github.com/maxi7524/pep-compass/tree/dev_max_integrated/docs)

Tutaj jest bałagan, jedynie tutoriale są zrobine. Mam do tego skillsy żeby to zrobić w [formacie Diataxis](https://diataxis.fr/#), oraz żeby to było generowane na githubie. Więc jak to zatweridzimyto po  porstu puszcze promta i będzie trzeba to potem jeszzce przejrzeć.

## Do przedyskutowania

Format notebooków - nie podoba mi się ten format - cięzko się w tym manewruje 

#TOOD - wymyślić nowy format 0- nadchodzaca analiza go ma to będzie można zobaczć 