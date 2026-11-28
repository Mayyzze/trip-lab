# Labelling conventions

One line of `dataset.jsonl` is a question, the date it is asked (`today`) and the expected
extraction. Labels are written by hand from the rules below, never by a model.

The block between the markers goes verbatim into the extraction prompt. Changing a rule means
checking every label again.

<!-- conventions:start -->
- city: the city name only, in its usual English spelling (e.g. "Lisboa" -> "Lisbon"), with typos
  corrected. An ambiguous name stays as written (e.g. "Paris"). null if no city is named, or if two
  cities are named.
- start, end: ISO dates, only if the user mentions a day or a period; otherwise both null, never
  guess. Resolve relative dates from today.
  - A single day: start and end are both that day.
  - A weekday name means its next occurrence, today included: "on Friday" asked on a Friday is today.
  - A date without a year means its next occurrence.
  - "This weekend" is the coming Saturday and Sunday. If today is Saturday, it is today and Sunday;
    if today is Sunday, it is today only.
  - "Next weekend" is the first weekend whose Saturday is after today: from Monday to Friday it is
    the same as "this weekend", on Saturday or Sunday it is the following one.
  - Extract dates however far in the future they are.
- topic: "weather" if they ask about the weather; "activities" if they ask what to see, do, eat or
  visit; "both" if they ask for the actual weather and for suggestions. A hypothetical condition,
  as in "what can I do if it rains?", is "activities".
- country: ISO 3166-1 alpha-2 code (e.g. "US", "FR") only if the user names a country or a state or
  region that identifies one (e.g. "Texas" -> "US"). A city name alone never sets it. Otherwise null.
- region: the state or region name in English, only if the user names one (e.g. "Texas");
  otherwise null.
<!-- conventions:end -->

## Tags

| Tag | Meaning |
|---|---|
| `relative_date` | the dates depend on `today` ("tomorrow", "this weekend", "on Friday") |
| `convention` | the label follows a choice made above that another team could make differently |
| `non_english` | the question is in French, Italian or Spanish |
| `no_date` | no day or period is mentioned |
| `ambiguous_city` | several places share the city name |
| `two_cities` | the question names two cities |
| `beyond_horizon` | the dates start after the last forecast day (today + 15) |
| `typo` | the question contains spelling mistakes |

No question reuses a city given as an example in the rules for the behaviour that example shows:
the prompt would contain the answer.
