# Vaults

A vault profile binds a Markdown vault path to a ProductiveBrain data directory and quarantine folder.

```bash
pb vault list
pb vault current
pb vault add main ~/brain
pb vault use main
pb vault rename old-name new-name
pb vault remove old-name
pb vault doctor
pb vault scaffold
```

Each profile has:

- a Markdown vault path
- a SQLite/runtime data directory
- a quarantine folder, usually `Learning/Inbox/pb`

Graph inspection commands help audit Markdown links:

```bash
pb vault graph
pb vault graph path/to/note.md
pb vault neighbors path/to/note.md
pb vault orphans
```

Use `pb --vault NAME ...` to run a single command against a non-active profile.
