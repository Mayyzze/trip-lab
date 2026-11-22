# Solution: TRIP-14 and review of `main.py`

Line numbers refer to `../main.py` as delivered.

## Issues

| # | Tag | Category | Line(s) | Issue |
|---|-----|----------|---------|-------|
| 1 | Blocking | Security | 99 | API key written to the logs |
| 2 | Blocking | Correctness | 85 | Days with a `0` value are dropped |
| 3 | Blocking | Quality | 93-94 | Every failure in the weather call is swallowed |
| 4 | Should fix | Correctness | 53 | Country code is never normalised |
| 5 | Should fix | Correctness | 54 | Region matched as a substring |
| 6 | Should fix | Scalability | 78-86 | One HTTP request per forecast day |
| 7 | Should fix | Scalability | 64, 69-75 | Unbounded, process-wide geocoding cache |
| 8 | Should fix | Structure | 108, 136 | Settings read from a global instead of the injected object |
| 9 | Should fix | Structure | 124 | Core service prints to stdout |
| 10 | Should fix | Quality | 104, 111 | Model output used without checking it |
| 11 | Nit | Quality | 18, 137-138 | `http_timeout_s` is dead, timeout and horizon are literals |
| 12 | Nit | Quality | 118-122 | Window rule inlined, two failures share one message |

### 1. API key in the logs (Blocking, line 99)
`logging.debug("... key=%s", ..., get_secret_value())` prints the clear-text key as soon as
someone runs `trip -v` and pastes the output into a ticket or CI log. `SecretStr` exists to
prevent exactly this.
**Fix:** never log the secret; log the model name only (see `MistralLLM.__init__` in `solution/main.py`).

### 2. Zero values dropped (Blocking, line 85)
`if all((t_min, t_max, precip, code))` treats `0.0` and `0` as missing. A dry day
(`precipitation_sum == 0.0`, i.e. most days), a day at 0 °C, or a clear sky
(`weather_code == 0`) disappears from the forecast. The model then says the day is
"missing", or `get_forecast` returns `[]` and the user gets "no forecast available".
The unit tests do not see it because they use a fake provider.
**Fix:** test for `None` explicitly: `if None in (t_min, t_max, precip, code)`. Covered by
`test_a_dry_freezing_clear_day_is_kept`.

### 3. Swallowed failures (Blocking, lines 93-94)
`except Exception: return {}` hides timeouts, 4xx/5xx, bad JSON and even programming errors,
and says nothing. Downstream, `geocode` returns `[]` so an outage becomes
`unknown city: 'Lisbon'`, and the forecast loop silently skips days. The user is told
something false and nobody can diagnose it.
**Fix:** catch `(httpx.HTTPError, ValueError)`, log a warning with the cause and raise
`WeatherUnavailable`. Covered by `test_a_failing_provider_is_reported_not_swallowed`.

### 4. Country not normalised (Should fix, line 53)
`c.country_code == country` is case-sensitive and `TripRequest` no longer cleans the value.
The model may return `"us"` or `" US"`, and then every candidate is filtered out:
`CityNotFound` for a city that exists.
**Fix:** normalise in a `field_validator` on `TripRequest.country` (strip, upper-case,
blank to `None`).

### 5. Region substring match (Should fix, line 54)
`region.casefold() in c.region.casefold()` makes "York" match "New York", "Georgia" match
"South Georgia...", etc., so the wrong namesake can be picked or a real ambiguity hidden.
**Fix:** compare with `==` on the case-folded values.

### 6. One request per day (Should fix, lines 78-86)
A 16-day forecast is 16 sequential round trips (about 16 x latency, and 16 chances to
hit the rate limit) for data Open-Meteo returns in one call with `start_date`/`end_date`.
**Fix:** one ranged request and `zip(..., strict=True)` over the returned columns.

### 7. Process-wide geocoding cache (Should fix, lines 64, 69-75)
`_geo_cache` is a module global with no size limit or expiry, keyed on the raw string
(`"Paris"` and `"paris"` are different entries), shared by every provider instance and
test, and it also caches the empty list produced by a failed call (see 3). In a long-lived
process it grows without bound and serves stale data.
**Fix:** drop it (the CLI makes one lookup per run), or if caching is wanted, make it
per-instance, bounded (`functools.lru_cache(maxsize=...)`), keyed on the normalised name,
and never cache failures.

### 8. Hidden global settings (Should fix, lines 108, 136)
`MistralLLM` receives `settings` but `write_answer` calls `get_settings()` instead, so the
model used by the answer step cannot be changed by what was injected (tests, other
configs), and `main` calls `get_settings()` twice.
**Fix:** use `self.settings` everywhere and read settings once in `main`.

### 9. Service prints (Should fix, line 124)
`print(f"resolved ...")` in `TripService.answer` writes to stdout before the answer, so
`trip "..." | pbcopy` or any script reading stdout gets an extra line, and the core can no
longer be reused outside a terminal. It also bypasses `-v`.
**Fix:** `logger.debug(...)`; only `main` prints.

### 10. Unchecked model output (Should fix, lines 104, 111)
`res.choices[0]` raises `IndexError` on an empty response, `parsed` may be `None` (then
`req.end` raises `AttributeError`), and `content` can be `None` or a list of chunks, which
`print` shows as-is. These are raw tracebacks instead of the `TripError` the CLI knows how
to display.
**Fix:** guard `choices`, `parsed` and `content` and raise `TripError` with a message.

### 11. Dead setting and literals (Nit, lines 18, 137-138)
`http_timeout_s` is defined but never read; `5.0` and `16` are hard-coded in `main`
(`forecast_horizon_days` was removed from `Settings`). Changing `.env` has no effect.
**Fix:** restore `forecast_horizon_days` and pass both values from `settings`.

### 12. Window rule inlined (Nit, lines 118-122)
The date-range rule lives inside `answer`, so it cannot be tested without a service, and
`end < start` and "in the past" produce the same message
("X to Y is not a valid range"), which does not tell the user what to fix.
**Fix:** a pure `check_forecast_window(req, today, horizon_days)` with one message per case.

## Reference implementation of TRIP-14

`main.py` in this folder (fixes above plus the feature) and `tests/` (35 tests, offline).
Run with `uv run --project .. pytest -q --rootdir . -c pytest.ini` from this folder.

- `parse_language` is the single validation rule (letters of any script, single space or
  hyphen between words, at most 30 chars). It backs both the argparse `type=` (exit code 2
  before any network call) and the `Settings.answer_language` validator (blank means unset,
  invalid fails at startup). Since the value ends up in the system prompt, the strict
  pattern is also what prevents prompt injection through `--lang` or `.env`.
- `resolve_language(cli_value, settings)`: `--lang` > `ANSWER_LANGUAGE` > `None`.
- `build_answer_prompt(language)`: replaces "answer in the language of the question" by
  "write the answer in X, whatever the language of the question or of the destination" when
  a language is set; units are unchanged. The extraction prompt is not touched.
- `TripService.answer(question, language=None)` and `write_answer(..., language=None)`
  carry the value through the existing seams. `FakeLLM` records `languages`.
