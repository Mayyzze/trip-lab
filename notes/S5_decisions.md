# S5 — journal des décisions

## 2026-11-28 — « next weekend »
Contexte : l'expression n'a pas de sens unique.
Décision : premier week-end dont le samedi est après today (= « this weekend » du lundi au vendredi).
Alternatives écartées : « this weekend » + 7 jours.
Conséquences : seuls les items posés un samedi ou un dimanche distinguent les deux expressions.

## 2026-11-28 — Deux villes, et jour nommé égal à today
Contexte : une seule ville dans le schéma ; « on Friday » posé un vendredi.
Décision : deux villes → city = null ; un jour nommé inclut aujourd'hui.
Alternatives écartées : première ville citée (l'autre serait ignorée sans le dire) ; +7 jours.
Conséquences : le service demande la ville quand il y en a deux.

## 2026-11-28 — country / region dans l'exact match
Contexte : le repo les extrait et `pick_location` filtre strictement dessus.
Décision : étiquetés (null par défaut) et comptés ; « A city name alone never sets it » ajouté.
Alternatives écartées : colonnes à part ; non évalués (un pays inventé ne coûterait rien).
Conséquences : « all fields » couvre les 6 champs ; règle de production précisée.

## 2026-11-28 — Typos
Contexte : le dataset contient « Edinburg » et « Marseile ».
Décision : la règle city dit « with typos corrected ».
Alternatives écartées : laisser « usual English spelling » l'impliquer.
Conséquences : le label corrigé découle d'une règle écrite.

## 2026-11-28 — Schéma du dataset dans `src/trip/dataset.py`
Contexte : le test et le script d'éval lisent le même fichier.
Décision : un seul `DatasetItem`, `extra="forbid"`, city/start/end sans valeur par défaut.
Alternatives écartées : schéma dans le test, recopié par le script.
Conséquences : une clé mal tapée ou absente fait échouer le chargement.

## 2026-11-28 — Pas de fuite des exemples de règles
Contexte : les règles citent « Lisboa », « Paris », « Texas » et iront dans le prompt.
Décision : aucune question ne teste un comportement avec la ville de l'exemple.
Alternatives écartées : garder « Paris, Texas » en test (le prompt contiendrait la réponse).
Conséquences : « Portland, Maine » et « London » remplacent ces cas.

## 2026-11-28 — `today` dans le message user
Contexte : chaque exemple few-shot a sa propre date.
Décision : « Today is <weekday> <date> » précède la question, exemples compris ; `EXTRACTION_RULES` est fixe.
Alternatives écartées : date dans le bloc système (deux formats différents pour le seul few-shot).
Conséquences : écart au brief ; la seule différence entre variantes reste la technique.

## 2026-11-28 — Une méthode `complete`, pas d'appel à `chat.parse`
Contexte : `parse_async` lève `JSONDecodeError` sur une sortie tronquée et perd tokens et finish_reason.
Décision : `complete(..., response_schema=)` envoie la même requête ; validation dans `extract.py`.
Alternatives écartées : `chat.parse` avec les échecs sans coût mesuré.
Conséquences : un échec `structured` a ses tokens ; `ChatModel` est un port séparé de `LLMClient`.

## 2026-11-28 — Latence et retries mesurés dans `llm.py`
Contexte : E8 exclut des percentiles les appels retentés ; aucun appel chat n'était retenté.
Décision : `@retry` sur `complete`, chronomètre autour de chaque tentative, `retries` dans `Completion`.
Alternatives écartées : chronométrer dans le script (compterait l'attente) ; modifier `retry.py`.
Conséquences : épuisement des 4 tentatives → `LLMUnavailable`, que le script pourra enregistrer.

## 2026-11-28 — Échec de format et parser
Contexte : E4 ; il faut garder le texte et les tokens d'un échec.
Décision : `parsed=None` (loggé) ; `length` et `model_length` échouent même si un JSON se lit ; `structured` validé strictement.
Alternatives écartées : lever l'erreur ; parser tolérant pour `structured` (masquerait un schéma ignoré).
Conséquences : le parser tolérant prend le dernier objet JSON lisible, même si un objet suivant est cassé.

## 2026-11-28 — Deux chemins d'extraction jusqu'à l'étape 5
Contexte : `TripService` utilise encore `parse_question`.
Décision : le garder jusqu'au choix de la variante.
Alternatives écartées : brancher tout de suite une variante provisoire, non mesurée.
Conséquences : deux prompts coexistent ; le bug de troncature reste en production d'ici là.

## 2026-11-28 — `n/30` avec plusieurs runs
Contexte : 3 runs par question, le brief demande des comptes sur 30 (E7).
Décision : juste si juste dans plus de la moitié des runs ayant répondu ; flip = la prédiction change.
Alternatives écartées : compte sur 90 appels ; premier run seul.
Conséquences : l'instabilité se lit dans la colonne flips, pas dans l'exact match.

## 2026-11-28 — Erreurs API et latence
Contexte : un 429 épuisé ne mesure pas le prompt (E4 ne couvre que le format).
Décision : exclu du jugement, compté dans `errors` ; question fausse sans aucune réponse ; p50/p95 sans retries.
Alternatives écartées : compter l'erreur comme une réponse fausse.
Conséquences : avec `errors` > 0, les variantes sont jugées sur des nombres de runs différents.

## 2026-11-28 — Ordre des appels et traces
Contexte : charge et rate limit varient pendant un run payant.
Décision : tâches par run, question, puis variante ; chaque trace écrite dès la réponse.
Alternatives écartées : variante par variante (latence biaisée) ; écriture finale (perdue si crash).
Conséquences : les lignes du fichier sont dans l'ordre d'arrivée.

## 2026-11-28 — Logique dans `src/trip/evaluation.py`
Contexte : les tests doivent importer scoring et boucle d'exécution.
Décision : le script ne fait que les options, l'assemblage et l'affichage ; normalisation sans `slugify`.
Alternatives écartées : tout dans le script ; `slugify` (supprime « Ł »).
Conséquences : `country`/`region` ont leurs colonnes ; `error_detail` garde la cause d'un appel sans réponse.

## 2026-11-28 — Variante en production : `few_shot`
Contexte : run `s5-20261128-093940`, mistral-small-latest, choix dans `notes/S5.md`.
Décision : `Settings.extraction_variant = "few_shot"`, injectée par `cli.py` via `VARIANTS`.
Alternatives écartées : construire l'extracteur dans `TripService`.
Conséquences : mesure valable pour ce modèle ; le défaut `mistral_model` reste `large`, non mesuré.

## 2026-11-28 — Suppression de `parse_question`
Contexte : deux chemins d'extraction depuis l'étape 2.
Décision : `parsed=None` → `TripError` ; panne → `LLMUnavailable` ; `EXTRACT_PROMPT` supprimé.
Alternatives écartées : repli sur une autre variante (enchaînement non mesuré).
Conséquences : bug de troncature disparu ; appels retentés ; `max_tokens` 200 → 300.
