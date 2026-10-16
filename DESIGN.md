# CrackRuler UI: design decision (locked)

**Subject world:** a field inspector standing at a wall with a phone, a printed scale marker and a
plastic *crack-width comparator card* (the transparent card with printed lines of 0.1, 0.2, 0.3 mm
that inspectors hold against a crack). Lumber-crayon marks next to cracks. Grey concrete.

## Directions considered
1. **Comparator card (chosen).** Light cured-concrete grey, condensed engineering display face, the
   comparator card itself as the signature: every measured width is plotted as a crayon tick on a
   strip of true-thickness reference lines with the exposure guideline notched in.
2. Night field tablet: graphite, hi-vis yellow, big touch targets, everything mono. Rejected: reads as
   a generic dark dashboard; ledger already holds several graphite + yellow rows (MESH-C2).
3. Engineering drawing sheet: title block, revision table, blueprint linework. Rejected: blueprint is a
   cliché attractor and fights the photo evidence.

## Tokens
- `--concrete` #D9DAD5 (light bg) / `--wet-concrete` #181A19 (dark bg, system preference)
- `--slab` surface shift ±4 %, `--ink` #1E2120 / #E4E5E0, `--ink-dim`
- `--crayon` #F2C200, used only as the *mark* (tick, highlight band under the verdict word, focus ring)
- verdict semantics shared with the evidence overlay: `--ok` #2E8B57, `--watch` #C98A00, `--over` #C2362F
- Type: **Saira Condensed** 300/700 (display: the condensed numerals of gauge cards and stencilled
  signage), **Public Sans** 400-800 (body: civic, very legible, US gov typeface), **Martian Mono**
  (every number with a unit). All OFL, self-hosted (CSP `default-src 'self'`).
- Radius language: 2px on controls, 0 on evidence and the comparator (printed objects have square corners).
- Separation: whitespace, then ±4 % slab shift; hairlines only on the comparator.

## Layout
Asymmetric 4/8 split: left = the inspector's three inputs (photo, exposure, marker); right = evidence
(verdict line, annotated photo, comparator, crack rows, agent trace). One primitive repeated: the
**ruled row** (label | value | mono unit), used for exposure table, crack table and trace.

## Signature elements
1. Comparator card strip with true-relative line thickness and the ACI guideline notch.
2. Crayon tick drawn onto the comparator for each crack (scaleX from the left, expo-out).
3. Exposure picker where each option shows its guideline width as a printed line.
4. Agent trace replayed at the real step timings (scaled), each step ruled like a log book.
5. Verdict word set huge in Saira Condensed with a crayon band behind it.

## Motion
Entrance: wordmark masked line reveal, inputs stagger 60 ms. Result: verdict band wipes in (clip-path),
comparator ticks draw with `cubic-bezier(.16,1,.3,1)` 700 ms, trace rows tick in at replayed timing.
Hover: row shifts 2px right with crayon left-edge; buttons press 1px down. `prefers-reduced-motion`
disables all of it.
