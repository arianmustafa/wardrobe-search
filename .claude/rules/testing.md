# Testing rules

<!-- Deliberately NOT path-scoped: the approval workflow below must be active
     before any test file exists or is read. -->

## Intent
- Never decide what "correct" means on your own. Ask for the behavior, or work
  from the scenarios you were given.
- Before writing code, propose behavior scenarios — grouped as happy path /
  edge / error / non-goals — and wait for the human to approve, cut, and add
  to them.

## Writing tests
- Test behavior, not implementation. A test describes what the system does,
  never how the code is structured inside. No calls to private helpers.
- Name the test after the behavior. The body is setup → action → expected result.
- A test must be able to fail. Write it first and watch it go red, or
  temporarily break the implementation to confirm it fails — a test that has
  never been seen failing proves nothing.
- Mock only what you don't own (network, clock, filesystem, third-party APIs).
  Let the real domain code run.
- Never write a test that just confirms the current (possibly buggy) behavior.
  If there's no approved spec, ask before asserting.

## Maintaining tests
- When a test fails, read the message and fix the code — don't edit the test to
  make it pass unless the expected behavior actually changed.
- If a refactor breaks a test but the behavior didn't change, the test was
  wrong — fix the test's coupling, don't chase the implementation.
- Keep failures specific: assert on meaningful values and messages, so the next
  failure says what broke, not just "expected true, got false."

## Structure
- Put shared setup behind helpers written in domain language; hide noise, never
  meaning.
- Split tests by feature or behavior, not into one giant file.
- Match the existing tests in `backend/tests/` — they are the template.

## How this suite implements the rules
- The approved behavior spec is encoded in `backend/tests/` (approved
  2026-07-17). Changing an assertion there is a spec change — confirm with the
  human first.
- The only mock is the third-party Gemini client: the `fake_gemini` fixture in
  `conftest.py` records calls and returns chosen vectors, so tests can assert
  on what the app sends (downscaled JPEG) and control search scores exactly.
  Everything else — image validation, ChromaDB, normalization — runs real.
- `conftest.py` redirects the data directories to a temp dir and blanks the API
  key env vars *before* importing app modules (the store opens ChromaDB at
  import time). Keep that ordering.
- New or changed tests get a mutation check: temporarily break the behavior in
  the app code, confirm the matching test goes red, restore, and report which
  mutations were caught.
