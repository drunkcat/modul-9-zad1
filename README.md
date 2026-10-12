# Prognoza czasu półmaratonu

Aplikacja szacuje wynik półmaratonu na podstawie płci, wieku oraz czasu uzyskanego na dystansie 5 km. Użytkownik opisuje się zwykłym tekstem, a model językowy zamienia wypowiedź na uporządkowane dane. Następnie wytrenowany model wylicza przewidywany czas.

## Jak działa rozwiązanie

- Wyniki zawodów z dwóch edycji półmaratonu (2023 i 2024) są przechowywane w DigitalOcean Spaces.
- Proces przygotowania danych łączy zestawy, oblicza wiek uczestników, konwertuje czasy na sekundy i usuwa rekordy z brakami lub niepoprawnymi wartościami.
- Testowane są trzy podejścia regresyjne: **regresja liniowa**, **Random Forest** i **Gradient Boosting**. Wyniki są porównywane za pomocą MAE, RMSE i R².
- Najlepszy wariant **Gradient Boosting** jest ponownie trenowany na pełnym przygotowanym zbiorze, zapisywany w formacie Joblib i przesyłany do Spaces.
- Interfejs odczytuje dane z opisu przy użyciu OpenAI, pobiera zapisany model z Spaces i pokazuje prognozę w formacie godzin:minut:sekund.

Prognoza jest orientacyjna. W eksperymencie średni błąd bezwzględny na zbiorze testowym wyniósł około 5 minut.

## Dane w DigitalOcean Spaces

Bucket powinien zawierać pliki w następujących lokalizacjach:

```text
data/halfmarathon_wroclaw_2023__final.csv
data/halfmarathon_wroclaw_2024__final.csv
models/halfmarathon_gradient_boosting_v1.joblib
```

Pliki CSV są odczytywane z separatorem średnika. Przy użyciu innych plików należy dopasować ich klucze obiektów oraz nazwy kolumn wykorzystywanych do budowy cech i celu. Wymagane kolumny źródłowe to `Płeć`, `Rocznik`, `5 km Czas` oraz `Czas`.

Dostęp do obiektów odbywa się przez API S3 z użyciem kluczy Spaces. Bucket i pliki mogą pozostać prywatne.

## Konfiguracja środowiska

Utwórz lokalny plik `.env` w katalogu projektu i uzupełnij wartości:

```dotenv
OPENAI_API_KEY=...
DO_SPACES_BUCKET=...
DO_SPACES_REGION=...
DO_SPACES_ENDPOINT=https://<region>.digitaloceanspaces.com
DO_SPACES_KEY=...
DO_SPACES_SECRET=...
```

`OPENAI_API_KEY` służy do odczytywania opisu użytkownika. Zmienne `DO_SPACES_*` konfigurują połączenie z bucketem i pobieranie modelu. Wartość regionu oraz endpointu muszą odpowiadać lokalizacji bucketa.

Nie umieszczaj prawdziwych kluczy w kodzie, notebookach ani publicznym repozytorium. Plik `.env` powinien pozostać lokalny i być wykluczony z kontroli wersji.

## Uruchomienie lokalne

1. Użyj wersji Pythona zgodnej z plikiem `.python-version`.
2. Zainstaluj pakiety wymienione w pliku zależności projektu.
3. Skonfiguruj `.env` zgodnie z powyższym przykładem.
4. Uruchom interfejs poleceniem Streamlit, wskazując skrypt aplikacji jako punkt wejścia.

## Trenowanie i publikacja modelu

Notebook treningowy wymaga poprawnego `.env` oraz dostępu do obu plików CSV w folderze `data/`. Po wykonaniu etapów przygotowania i oceny zapisuje model lokalnie w katalogu `models/`, a następnie przesyła go do bucketa pod ścieżkę pokazaną wyżej. Interfejs pobiera właśnie ten obiekt, dlatego po zmianie nazwy lub lokalizacji modelu trzeba zaktualizować konfigurację po stronie aplikacji.

Do trenowania klucz Spaces musi umożliwiać odczyt danych i zapis modelu. Dla samego działania interfejsu wystarczy osobny klucz z prawem odczytu do bucketa.

## Główne biblioteki

- Streamlit — interfejs aplikacji.
- OpenAI i Pydantic — ekstrakcja oraz walidacja pól z wypowiedzi.
- boto3 — komunikacja z DigitalOcean Spaces przez API S3.
- scikit-learn — przygotowanie, trenowanie i ocena modeli regresyjnych.
- Joblib — zapis oraz odczyt wytrenowanego modelu.
- pandas — przygotowanie tabel danych wejściowych i treningowych.
