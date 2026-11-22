# TRIP-12: Imperial units option

**Type:** Feature  **Area:** CLI / answer generation

## Context
Travellers from the US and UK get forecasts in °C and mm, which they have to convert
by hand. Add a `--units` option so the answer is given in °F and inches.

## Acceptance criteria
1. `trip --units imperial "<question>"` is accepted. The default is `metric`, so
   `trip "<question>"` behaves exactly as before. Any other value (e.g. `kelvin`) is
   rejected by the argument parser with exit code 2.
2. With `imperial`, temperatures are converted from °C to °F (`C * 9/5 + 32`,
   rounded to 1 decimal) and precipitation from mm to inches (`mm / 25.4`, rounded to
   2 decimals) before reaching the model. The model must not convert anything itself.
3. The answer prompt and the data sent to the model name the unit system in use
   (°F / in or °C / mm), so the written answer never mixes systems.
4. Metric output is unchanged: same values, same units, no regression for existing
   calls of `answer(question)`.
5. The conversions and the CLI option are covered by tests that run without network
   access or an API key.
