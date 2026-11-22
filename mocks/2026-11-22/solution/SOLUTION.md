# Review notes

Line numbers refer to the files under `src/trip/` of the mock.

## Blocking

1. **`models.py:41` single-day trips rejected (correctness).** `end <= start` fails
   when start == end ("what's the weather tomorrow?"), the most common question. The
   error message even says "before". Fix: `self.end < self.start`.
2. **`models.py:46` horizon off by one (correctness).** `today + horizon_days` allows a
   17th day. Open-Meteo then returns a 400 and the user sees "weather service
   unavailable" instead of a date error. Fix: `timedelta(days=horizon_days - 1)`.
3. **`weather.py:55-58` missing data turned into zeros (correctness).** `x or 0`
   replaces `None` with 0 °C / 0 mm / weather code 0 (clear sky). The model then states
   invented figures as fact. Fix: skip the day when any value is `None` and log it
   (original behaviour), so the prompt's "day is missing" rule applies.
4. **`chat.py:41` API key written to logs (security).** `-v` prints the Mistral key
   (and the raw question) to stderr, where it ends up in terminal history and CI logs.
   Fix: drop the key from the message; never call `get_secret_value()` outside client
   construction.
5. **`weather.py:70` no timeout and a new client per call (scalability).**
   `timeout=None` lets one stalled request hang the process forever, and a fresh client
   means a new TCP/TLS handshake for each of the two calls. Fix: one module-level
   `httpx.Client(timeout=settings.http_timeout_s)`.

## Should fix

6. **`weather.py:15,37,62` unbounded, never-expiring cache (scalability).** Keyed on
   coordinates and dates, grows with every distinct request and serves stale
   forecasts indefinitely in a long-lived process. Fix: remove it, or use a bounded TTL
   cache (`cachetools.TTLCache`).
7. **`chat.py:39-90` one 50-line function (structure).** Parsing the question, window
   check, geocoding, forecasting and answer writing are interleaved, which makes each
   step untestable alone and is why the feature is awkward to add. Fix: split back into
   `_parse_request` and `_write_answer`.
8. **`weather.py:35-63` forecasts returned as untyped dicts (structure).**
   `DailyForecast` exists (and is imported, unused) but the contract between
   `weather` and `chat` is now string keys, with `day` an ISO string instead of a
   `date`. Typos fail silently at runtime. Fix: build `DailyForecast` objects.
9. **`chat.py:89-90` blanket `except Exception` returning raw JSON (quality).** An API
   failure, bad key or the empty-answer `TripError` is swallowed and the user is handed
   a JSON dump with exit code 0. Fix: remove the try/except; let `TripError` through and
   wrap SDK errors in `TripError`.
10. **`weather.py:61` `print` on stdout (quality).** Pollutes the program's output
    (`trip ... > out.txt` contains the log line). Fix: `logger.debug`.

## Nit

11. **`weather.py:18-19` `max_results` ignored (quality).** Count is hard-coded to 10
    although only the first hit is used. Fix: `"count": max_results`.
12. **`weather.py:7,74` leftovers.** `DailyForecast` imported but unused, logging done
    with an f-string (eager formatting). Fix: use lazy `%s` formatting; drop or use the
    import. Also the dropped type hints on `geocode`, `answer`.

## Reference implementation of TRIP-12

Under `solution/src/trip/` (clean baseline of all the above plus the feature):

- `units.py`: `c_to_f`, `mm_to_in`, `convert_forecast`, `LABELS`.
- `chat.py`: `answer(question, units="metric")`; the prompt is formatted with the unit
  labels and the payload carries `units` and converted values.
- `cli.py`: `--units {metric,imperial}`, default `metric`.
- `solution/tests/test_units.py`: run with
  `PYTHONPATH=solution/src uv run pytest solution/tests` (set `MISTRAL_API_KEY=x`).
