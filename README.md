# Prognoza czasu półmaratonu

To moja pierwsza aplikacja zbudowana w Streamlit i wdrożona na DigitalOcean. Pomaga oszacować czas ukończenia półmaratonu na podstawie krótkiego opisu biegacza.

## Jak korzystać z aplikacji

Wpisz w polu tekstowym swoją płeć, wiek i typowy czas biegu na 5 km. Możesz napisać to zwyczajnie, na przykład: „Mam 34 lata, jestem kobietą i przebiegam 5 km w 26 minut”.

Aplikacja odczyta te trzy informacje. Jeśli którejś zabraknie, pokaże, co trzeba uzupełnić. Gdy dane są kompletne, model oszacuje czas półmaratonu i wyświetli go w godzinach, minutach i sekundach.

## Skąd bierze się prognoza

Model został wytrenowany na historycznych wynikach półmaratonu z lat 2023 i 2024. W czasie przygotowania danych czasy zostały zamienione na sekundy, obliczono wiek zawodników i pominięto niekompletne lub nieprawidłowe wpisy. Porównano kilka modeli regresyjnych, a do prognoz wybrano Gradient Boosting.

Pliki z wynikami oraz gotowy model są przechowywane w DigitalOcean Spaces. Dzięki temu aplikacja może pobrać model, kiedy go potrzebuje.

Prognoza jest przybliżeniem, a nie gwarancją wyniku. W testach średnia różnica między przewidywanym a rzeczywistym czasem wynosiła około 5 minut. Rzeczywisty rezultat zależy też od treningu, trasy, pogody i samopoczucia w dniu biegu.

## Do czego służy Langfuse

Langfuse pomaga sprawdzać, jak działa część aplikacji korzystająca z modelu językowego. Zapisuje informacje o wywołaniach, takie jak użyty model, czas odpowiedzi i zużycie tokenów. Aplikacja dodaje także dwie proste oceny:

- **kompletność danych** — jaka część wymaganych informacji została odczytana;
- **podstawowa poprawność** — czy odczytane wartości mieszczą się w przyjętych zakresach.

Te oceny nie mówią jeszcze, czy LLM zawsze poprawnie zrozumiał wypowiedź. Do sprawdzenia dokładności potrzebny byłby zestaw przykładów z poprawnymi odpowiedziami przygotowanymi wcześniej.

Treść opisu użytkownika i odpowiedź LLM są maskowane przed wysłaniem śladu do Langfuse. W śladzie pozostają podstawowe informacje potrzebne do monitorowania działania ekstrakcji.
