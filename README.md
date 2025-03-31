# Aplikacja optymalizująca belki żelbetowe

Aplikacja ma na celu wykorzystanie sieci neuronowej w celu optymalizacji prekrojów pod względem ceny.
Program będzie optymalizował przekroje prostokątne i o krztałcie litery T.

## Spis treści

- [Instalacja](#instalacja)
- [Użytkowanie](#użytkowanie)
- [Funkcje](#funkcje)

## Instalacja

Trzeba odpalić plik main.py.

## Użytkowanie

Aplikacja jest napisana za pomocą biblioteki PyQt6. GUI jest podzielone na zakładki, w każdej zakładce będą prowadzone osobne obliczenia.

Plan zakładek:

1 - Ogólny opis programu

2 - Prosty kalkulator do wyznaczania momentów zginających.
    W planach:
    - dodanie możliwości pokazania wykresu sił wewnętrznych.
    - Dodanie możliwości dodania siły skupionej, na razie tylko obciążenie równomiernie rozłożone.

3 - Prosty kalkulator do wyznaczania zbrojenia w przekrojach (chyba to usunę).

4 - (w planach) Zakładka gdzie wpisujemy parametry i sieć nam optymalizuje przekrój prostokątny.

5 - (w planach) Zakładka gdzie wpisujemy parametry i sieć nam optymalizuje przekrój o krztałcie T.

6 - (w planach) Końcowa zakładka z credits itp.

## Funkcje

### GUI
Wszystkie zakładki są w folderze GUI\tabs, obliczenia które prowadzi się w zakładkach są w folderze calculations.

### ŚN
Dataset używany do trenowania modeli znajduję się w folderze datasets.

Kod użyty do wygenerowania datasetu znajduję się w folderze calculations\dataset\generate_dataset

Modele sieci neuronowych wraz ze skalarami są w folderze nn_models, odpowiednio dla przekrojów. Np. dla przekrojów
prostokątnych, znajdują się w nn_models\nn_models_rect_section. W tym folderze znajduję się także folder nn_models\nn_models_rect_section\_evaluate_model
w którym są skrypty, w których można przetestować jak precyzyjny jest model, dla poszczególnego feature.

W folderze neural_netoworks\rect_section\networks znajduję się kod trenujący modele a w neural_netowrks\rect_section\optuna znajduję się kod do znajdywania
hiperparametrów dla modelu.

W pliku calculations\optimization\optimize_rect_section.py jest końcowa optymalizacja przekroju prostokątnego (w takcie pracy)
