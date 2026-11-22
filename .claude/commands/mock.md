---
description: Build a broken copy of trip-lab modules for a review + improve mock
argument-hint: <paths...> -- <feature to add>
---
Arguments: $ARGUMENTS (paths before "--", the feature after it).
Copy the listed trip-lab paths into mocks/<date>/ as a self-contained uv
project
(include existing tests). If the only path is "monolith", flatten the
whole package into a single 150-line main.py instead. Degrade the code with 8
to 12 realistic issues:
3 correctness, 2 async or scalability, 2 structure, 1 security, the rest
quality.
No hint in names or comments. The happy path must still run.
Write FEATURE.md: the feature as a ticket with 3 to 5 acceptance criteria.
In solution/: SOLUTION.md (issues tagged Blocking / Should fix / Nit, with
line,
impact and fix) and a reference implementation of the feature.
Do not summarise any of this in your reply.