# TRIP-14: Answer in a configured language

**Type:** Feature  **Area:** CLI / settings / answer generation

## Context
Today the answer comes back in the language of the question. A team that asks in
English but shares forecasts with Italian colleagues (or someone who types short
questions in one language and wants to read in another) has to translate by hand.
Let the user fix the answer language once in configuration, and override it per call.

## Acceptance criteria
1. A new optional setting `answer_language` is read from `ANSWER_LANGUAGE` (env or
   `.env`). When it is unset or blank, behaviour is exactly as today: the answer is
   written in the language of the question.
2. `trip --lang Italian "<question>"` overrides the setting for that call. Precedence:
   `--lang` > `ANSWER_LANGUAGE` > language of the question.
3. A language value is a plain language name or code: letters (any script), single
   spaces or hyphens between words, at most 30 characters after trimming (e.g.
   `Italian`, `it`, `Brazilian Portuguese`). Anything else (digits, punctuation,
   newlines, an empty `--lang ""`, longer values) is rejected with exit code 2 before
   any network or model call. For the setting, an invalid value fails at startup.
4. When a language is set, the answer prompt tells the model to write the answer in
   that language whatever the language of the question or of the destination. Figures
   and units (°C, mm) are untouched. The extraction step (question to city and dates)
   is not affected, so a question in French about Lisbon still works with `--lang Italian`.
5. The language travels through the existing ports (`TripService.answer` and
   `LLMClient.write_answer` gain an optional `language`), `answer(question)` without it
   behaves as before, and everything above is covered by tests that run without network
   access or an API key.
