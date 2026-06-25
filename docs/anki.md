# Anki

ProductiveBrain uses local candidate state plus `genanki` export. AnkiConnect is optional.

## Flow

```bash
pb anki generate "Bayes theorem" --model flash-lite
pb anki pending
pb anki list --suggested
pb anki accept
pb anki reject
pb anki export
```

Candidate states are stored in SQLite:

- `suggested`: generated but not accepted
- `accepted` or `edited`: ready to export
- `rejected`: kept out of export
- `exported`: packaged or synced

`pb anki export` packages accepted/edited cards into `.apkg` by default. Use `pb anki export --csv` when you want CSV and no AnkiConnect path.

`pb doctor` may warn when AnkiConnect is offline. That does not block `.apkg` or CSV export.
