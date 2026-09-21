# Analiza 0_01

## MEROPS — ile mamy próbek, które weryfikują cięcie

Niektóre peptydazy mają bardzo mało cięć, trzeba by te peptydazy albo usunąć z próby, albo zweryfikować, z czego to wynika. Osobiście uciąłbym wszystko poniżej 10, żeby nie wyciąć za dużo peptydaz.
#TODO - zweryfikowac z Pauliną

### Komentarze:
* Informacja o liczbie cięć pochodzi z lokalnego MEROPS dataset 35. Dla każdej peptydazy `n_cleavages` jest liczbą zarejestrowanych przez MEROPS cięć substratów użytych do zbudowania jej profilu specyficzności. Panel obejmuje 55 wpisów MEROPS.
* Poszczególne proteazy są wpisami z panelu human serum MEROPS. Każdy wpis odpowiada osobnej peptydazie/profilowi specyficzności; na tym etapie nie interpretujemy jeszcze liczby cięć jako aktywności enzymu ani szybkości reakcji.
* Na wykresie każdy poziomy słupek odpowiada jednej proteazie. Oś X pokazuje `n_cleavages` w skali logarytmicznej. Proteazy są uporządkowane według liczby cięć. Linie przy 20 i 100 są jedynie progami referencyjnymi. Mediana panelu wynosi 20 cięć; 38/55 proteaz ma co najmniej 10 cięć, a 16/55 co najmniej 100.

## Zliczenia pozycji brzegowych
Widzimy, że brzegowe pozycje są wybrakowane, nie aż tak bardzo, ponieważ to tylko jakieś $30%$ próbek wybrakowanych na końcach, ale to i tak jest dosyć dużo. Widzimy, że miejsca przed cięciem są mniej wybrakowane i stanowią sensowniejszą pulę.

Warto uwzględnić w eksperymentach.

### Komentarze:
* `pooled cleavages` oznacza całkowitą liczbę cięć przypisaną danej proteazie, bez rozróżnienia, czy dla konkretnego cięcia dostępne były wszystkie pozycje $P_4,\ldots,P_4'$. Jest więc górnym ograniczeniem liczby obserwacji dla każdej konkretnej pozycji.
* Dla proteazy $\pi$ i pozycji $p$ liczba obserwacji wynosi:

$$
n_{\pi,p}=\sum_{a\in\mathcal A} C_{\pi,p,a},
$$

gdzie $C_{\pi,p,a}$ jest liczbą wystąpień aminokwasu $a$ na pozycji $p$. Następnie wykres używa względnej głębokości:

$$
r_{\pi,p}=\frac{n_{\pi,p}}{n_{\pi,\mathrm{pooled}}}.
$$

* Lewy wykres pokazuje rozkład $r_{\pi,p}$ dla każdej pozycji $P_4,\ldots,P_4'$. Wartość $1$ oznacza, że dana pozycja była obserwowana dla wszystkich cięć tej proteazy. Prawy wykres porównuje całkowitą liczbę cięć proteazy z liczbą obserwacji jej najgorzej pokrytej pozycji. Linia $y=x$ oznaczałaby pełne pokrycie wszystkich pozycji.

## Ilość informacji
Widzimy, że większość wyników jest specficzna, chociaż jest kilak która ma gorszy wynik nizten rozkąłd średni. 

### Komentarze 


Information content mierzy, jak bardzo rozkład aminokwasów na danej pozycji odbiega od rozkładu maksymalnie nieokreślonego. Dla 20 aminokwasów:

$$
IC(p)=\log_2 20-H(p),
$$

gdzie

$$
H(p)=-\sum_{a=1}^{20}p_a\log_2p_a.
$$

$IC=0$ oznacza brak preferencji między aminokwasami. Maksimum:

$$
IC=\log_2 20\approx4.32\ \text{bit},
$$

otrzymujemy wtedy, gdy na danej pozycji zawsze występuje dokładnie jeden aminokwas.

Przy małej liczbie obserwacji dodatni $IC$ może powstać wyłącznie przez sampling noise. Dlatego notebook dla każdej rzeczywistej liczby obserwacji symuluje próbki z rozkładu jednostajnego i wyznacza oczekiwany poziom:

$$
IC_{\mathrm{null}}(n).
$$

Ostatecznie interesuje nas:

$$
IC_{\mathrm{excess}}
=
IC_{\mathrm{observed}}
-
IC_{\mathrm{null}}(n),
$$

czyli ilość informacji ponad to, czego można oczekiwać wyłącznie z małej próby.

## Reprodukowalność wyniku

Widzimy, że część wyników jest ciężko reprodukowalna. Dla proteaz z bardzo małą liczbą obserwowanych cięć korelacja rankingu spada do około $0.3-0.5$. Nie jest to bardzo wysoka zgodność i oznacza, że ranking sekwencji może istotnie zmieniać się przy niewielkiej zmianie danych wejściowych. Dla proteaz z większą liczbą obserwacji stabilność jest znacznie większa; 11 proteaz osiąga medianę $\rho\geq0.95$.

### Komentarze

* Dla każdej proteazy i każdej pozycji $P_4,\ldots,P_4'$ zaczynamy od liczby obserwacji, która rzeczywiście występowała w MEROPS. Jeżeli dla danej pozycji było $n_{\pi,p}$ obserwacji, to każdy bootstrap również zawiera dokładnie $n_{\pi,p}$ obserwacji. Nie zwiększamy więc sztucznie wielkości próby.

* Z oryginalnych zliczeń wyznaczamy wygładzony rozkład aminokwasów:

$$
\hat p_{\pi,p,a}
=
\frac{c_{\pi,p,a}+0.5}
{\sum_b c_{\pi,p,b}+20\cdot0.5}.
$$

Następnie dla każdej pozycji niezależnie losujemy nowe zliczenia:

$$
C_{\pi,p}^{*(b)}
\sim
\operatorname{Multinomial}
\left(
n_{\pi,p},
\hat p_{\pi,p}
\right).
$$

Czyli jeżeli oryginalnie dla $P_1$ było 30 obserwacji, to również w każdym bootstrapie losujemy dokładnie 30 obserwacji, tylko ich rozkład pomiędzy 20 aminokwasów może być nieco inny.

* Taką procedurę wykonujemy dla wszystkich ośmiu pozycji i otrzymujemy nową bootstrapową macierz specyficzności $M_\pi^{*(b)}$. Kod zachowuje osobną rzeczywistą głębokość każdej pozycji; nie zakłada, że wszystkie pozycje mają tyle samo obserwacji. Sam bootstrap zliczeń jest wykonywany niezależnie dla każdego `(protease, subsite)`.

* W notebooku generujemy **32 takie bootstrapowe macierze** dla każdej proteazy (`N_BOOTSTRAP = 32`).

* Następnie oryginalną macierzą MEROPS oceniamy stały zestaw **1000 sekwencji testowych** i dostajemy:

$$
S_\pi^{orig}(x_1),\ldots,S_\pi^{orig}(x_{1000}).
$$

Każdą z 32 macierzy bootstrapowych wykorzystujemy do oceny dokładnie tych samych 1000 sekwencji:

$$
S_\pi^{*(b)}(x_1),\ldots,S_\pi^{*(b)}(x_{1000}).
$$

* Dla każdego bootstrapu liczymy jedną korelację Spearmana:

$$
\rho_\pi^{(b)}
=
\operatorname{Spearman}
\left(
S_\pi^{orig},
S_\pi^{*(b)}
\right).
$$

Czyli dla jednej proteazy otrzymujemy finalnie:

$$
\rho_\pi^{(1)},\rho_\pi^{(2)},\ldots,\rho_\pi^{(32)}.
$$

Nie opieramy więc wniosku na jednym pojedynczym resamplingu.

* Na wykresie pokazana jest **mediana z tych 32 korelacji**:

$$
\widetilde{\rho}_\pi
=
\operatorname{median}
\left(
\rho_\pi^{(1)},\ldots,\rho_\pi^{(32)}
\right).
$$

Dodatkowo notebook zapisuje 10. percentyl:

$$
\rho_{\pi,0.10},
$$

czyli wartość, poniżej której znajduje się około 10% bootstrapów. Jest to prosta informacja o dolnym ogonie stabilności.

* To nie jest test $p$-value. Bootstrap nie testuje tutaj hipotezy typu „czy korelacja jest różna od zera”. Służy do propagacji niepewności wynikającej z ograniczonej liczby obserwacji MEROPS. Pytanie brzmi:

  > jeżeli ponownie uzyskalibyśmy tyle samo obserwacji, ale ich skład aminokwasowy zmieniłby się zgodnie z niepewnością próbkowania, czy te same 1000 sekwencji nadal zostałoby uporządkowanych podobnie?

* Numerycznie dla najsłabiej określonych proteaz otrzymujemy np.:

  * `S01.191`: mediana $\rho=0.311$,
  * `S01.199`: $\rho=0.360$,
  * `S01.228`: $\rho=0.373$,
  * kilka kolejnych proteaz znajduje się w zakresie około $0.44-0.52$.

  Z kolei dobrze pokryte proteazy osiągają wartości bliskie $1$, np. $0.98-0.999$.

Schemat całej procedury jest więc:

$$
\text{oryginalne counts}
\rightarrow
\hat p
\rightarrow
32\times\text{bootstrap counts o tej samej liczebności}
\rightarrow
32\times M^*
\rightarrow
32\times\text{score 1000 tych samych sekwencji}
\rightarrow
32\times\rho_{\mathrm{Spearman}}
\rightarrow
\operatorname{median}(\rho).
$$


## Redundancja proteaz 

Na heatmaie są pokazane korelacje pomiedzy wynikami próbami (dbaasp) ocneian tymi macieramai i czy są one podobne. Widzimy że nie są one redundante 

*** 

## Podsumowanie
MEROPS nie jest idealny, żeby analiza miała sens musimy przefiltrować peptydazy. 

### Decyzja 
Z powodu koniecnzości reprodukoalnosc te sekwencje któe schodza poniżej median spearmana $0.05$ bym odrzucił, ponieważ nie są one wiarygodne. Żeby nie stracić za dużo próbek odrzucam te 5 zaznaczonych na plotach
