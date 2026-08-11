# Task: Update the Obsidian vault after your work

You have the Obsidian MCP server connected (tools: `obsidian_*`). After
implementing and testing your kernels, document the work in the vault.

## What to write
1. **Journal entry** — `journal/YYYY-MM-DD-chore-torch.md` (today's date). Include:
   - What you built (files, functions, kernel list) and on which branch/commit
   - The storage contract you implemented against (packed int32 words, 16
     codes/word little-endian, int16 counter, FP16 I/O) — note ANY deviation
   - **Semantic gaps you had to infer** from stubs/drifted code (exact
     threshold comparison, counter-reset timing, dW accumulation dtype) —
     these go in the journal AND in `memory/decisions.md`
   - Test results: exact pass/fail, measured errors, shapes covered
   - Perf notes if measured (ms/step, GFLOPS)
   - Link to the branch: `chore/torch`, commit hashes
   - Link to the previous journal entry (check `journal/JOURNAL.md` for the
     latest) and to `projects/ultimate-ai-model/ultimate-ai-model.md`
2. **Project note** — append a dated section to
   `projects/ultimate-ai-model/ultimate-ai-model.md` summarizing the port.
3. **Decisions** — append to `memory/decisions.md` any inference where the
   CUDA code was ambiguous (one entry per decision, with rationale).

## Conventions (follow exactly)
- Frontmatter on every note:
  ```yaml
  ---
  tags: [project/ultimate-ai-model, journal]
  date: YYYY-MM-DD
  status: active
  parent: journal
  aliases: []
  ---
  ```
- Use [[wikilinks]] — never bare paths — to link related notes.
- Vault paths: the Obsidian vault root differs from the git repo; address
  notes by vault-relative path via the MCP tools (e.g.
  `journal/2026-08-11-chore-torch.md`). If unsure, `obsidian_search_notes`
  first, then `obsidian_get_note` (format: "document-map") to find patch
  targets before `obsidian_patch_note` / `obsidian_append_to_note`.
- Do NOT write HANDOFF.md / PLAN.md / commit logs into the vault.
- If you created new notes, also append them to the relevant hub index
  (`journal/JOURNAL.md` for journal entries).

## Acceptance
A human can open the vault and, from your journal entry alone, understand:
what the port covers, the contract, what you had to infer, and how to verify
it. Report the exact vault paths you wrote in your final summary.
