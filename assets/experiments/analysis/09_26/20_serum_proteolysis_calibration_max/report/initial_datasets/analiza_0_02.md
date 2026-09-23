# Analiza 0_02

## Zliczenie egzopeptydaz oraz endopeptydaz

Widzimy, że większość peptydaz jest dobrze udokumentowana, jednak warto zwrócić uwagę, że egzopeptydazy trzeba koniecznie inaczej analizować, ponieważ generują niedozwolone miejsca cięcia.

Widzimy, że bez maski dostajemy ocenę dodatkowych $19.2%$ pozycji, które są niedozwolone — trzeba to uwzględnić.

### Komentarze

* Mapujemy MEROPS ID do UniProt. Z UniProt pobieramy `accession number` oraz `EC number`, a na podstawie `EC number` otrzymujemy klasę peptydazy. Definiujemy takie mapowanie:

```text
3.4.11    -> aminopeptidase
3.4.14    -> dipeptidyl-peptidase
3.4.15    -> peptidyl-dipeptidase
3.4.16-18 -> carboxypeptidase
3.4.21-25 -> endopeptidase
```

Jeżeli nie da się jednoznacznie ustalić geometrii z `EC number`, wykorzystywana jest nazwa proteazy. Jeżeli również ona nie pozwala na jednoznaczne przypisanie, geometria pozostaje `unknown` — nie zakładamy domyślnie, że jest to endopeptydaza.

* Niedozwolone pary oznaczają pary `(proteaza, wiązanie)`, które obecny `CleavagePotential` punktuje, mimo że dane wiązanie nie może być miejscem zdarzenia dla mechanizmu tej proteazy.

Dla peptydu długości $L$ istnieje $L-1$ wiązań peptydowych. Obecna implementacja dla każdej proteazy generuje wszystkie $L-1$ kandydatów.

Dla endopeptydazy jest to poprawne: potencjalnym zdarzeniem może być każde wewnętrzne wiązanie.

Dla egzopeptydazy przestrzeń zdarzeń jest inna. Dopuszczalne jest tylko określone zdarzenie terminalne, np.:

* aminopeptydaza — cięcie od N-końca,
* carboxypeptidase — cięcie od C-końca,
* dipeptidyl-peptidase — usunięcie odpowiedniego N-terminalnego dipeptydu.

Przykładowo dla peptydu długości $L=17$ mamy $16$ wiązań. Dla DPP4 obecny kod punktuje wszystkie $16$, ale tylko jedno zdarzenie jest geometrycznie dopuszczalne. Pozostałe $15$ nie są słabymi kandydatami ani negatywnymi przykładami — są zdarzeniami niedozwolonymi i nie powinny w ogóle trafiać do modelu.

* Te $19.2%$ nie oznacza, że $19.2%$ danych eksperymentalnych jest błędnych. Oznacza to, że $19.2%$ par `(proteaza, wiązanie)` wygenerowanych przez obecny sposób scoringu nie powinno istnieć w przestrzeni kandydatów.

* Liczenie wykonano na `probe peptides`: 1000 unikalnych peptydów DBAASP długości 10–25 aa. Nie jest to zbiór walidacyjny. Jest to próbka użyta do zmierzenia, jak duża część obecnie generowanej przestrzeni `(proteaza, wiązanie)` jest geometrycznie błędna.

Dla każdego peptydu długości $L$ aktualna implementacja generuje:

$$
(L-1)\cdot 55
$$

par `(proteaza, wiązanie)`.

Po zsumowaniu dla 1000 peptydów:

$$
N_{\mathrm{scored}} = 869935.
$$

Następnie dla każdej proteazy zastosowano ograniczenie wynikające z jej geometrii i policzono wyłącznie dopuszczalne zdarzenia:

$$
N_{\mathrm{allowed}} = 702948.
$$

Stąd:

$$
N_{\mathrm{forbidden}}
=
N_{\mathrm{scored}}-N_{\mathrm{allowed}}
=
869935-702948
=
166987.
$$

Zatem:

$$
\frac{N_{\mathrm{forbidden}}}{N_{\mathrm{scored}}}
=
\frac{166987}{869935}
\approx
0.192,
$$

czyli około $19.2%$ obecnie punktowanych par jest niedozwolonych.

* To liczenie można wykonać dla wszystkich analizowanych peptydów, a nie tylko dla 1000 `probe peptides`. Wtedy dostaniemy dokładny udział niedozwolonych par dla rzeczywistego zbioru, zamiast oszacowania na próbce.

Schemat liczenia dla pełnego zbioru:

```python
scored_pairs = 0
allowed_pairs = 0

for sequence in all_sequences:
    length = len(sequence)
    n_bonds = length - 1

    scored_pairs += n_bonds * len(panel_codes)

    for code in panel_codes:
        geometry = EventGeometry(geometry_by_code[code])
        allowed_pairs += int(
            allowed_cut_positions(length, geometry).sum()
        )

forbidden_pairs = scored_pairs - allowed_pairs
forbidden_fraction = forbidden_pairs / scored_pairs
```

## Zaburzenie
Widzimy że istnieje różnica miedzy wynikami. Wprowadzanie tła zaburza jednoznacznie wyniki. 

### Komentarz
- lewy wykres pokazuje jak przy zakrywaniu kolejnyhc pozycji powstaej róznica score'u
- prawy pokazuej jak to ma siędo oryginalnego score'a 

## Wnioski

### Decyzje

Trzeba **koniecznie** zastosować maski dla egzopeptydaz. Stosowanie tła róznicuje wyniki. Uznaje to za artefakt, wstawiam tam zera (żeby endopeptydazy były karane z próbę wybrania zewnętrznych fragmentów) - pytanie czy to wogóle ejst dozowolen czy trzeba to wyciąć. 

#TODO - dopytac się Pauliny odnośnie czy to trzeba odrzucić czy akceptujemy


