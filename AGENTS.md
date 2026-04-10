# AI4LT Workspace Skills

This file is the local skill registry for this repo.
Before answering a homework request, load this file first.

## Invocation Format

Use one of these forms when you want to trigger a skill:

- `/lt-proof write a solution for the prompted question.`

Preferred convention: start the request with `/skill-name` and then state the task.

## Shared Defaults

When a skill in this file is invoked:

1. Read the referenced problem statement from the repo before solving.
2. If the homework is only available as a PDF and no companion statement file exists yet, create a reusable LaTeX transcription such as `hw_1_statement.tex` in the same homework folder.
3. Prefer reusing the companion statement `.tex` file on later turns instead of re-reading the PDF.
4. Separate brainstorming from the final polished solution.
5. State assumptions when the problem statement or prior work is ambiguous.
6. Prefer concise, checkable arguments over long intuitive explanations.
7. Write final proof-based homework answers in LaTeX unless the user asks for plain text.
8. When formatting a final writeup, use `AI4LT/assets/tex/hw_template.tex` as the base template unless the user says otherwise.
9. Mirror the assignment PDF structure whenever possible, using section and subsection titles that match the problem set.

## Sample Skill: `lt-proof`

### Purpose

Help with learning theory proof problems by turning a problem statement into a structured proof plan, a draft argument, and a polished answer.

### Use When

Use this skill when the user asks for any of the following:

- solve or start a proof-based learning theory question
- break a theorem or claim into smaller proof steps
- turn informal reasoning into a clean homework solution
- check whether a proof sketch is missing assumptions or logical steps

### Inputs

Expect some or all of the following:

- a homework reference such as `AI4LT/hw1/hw_1.pdf`
- a companion statement file such as `AI4LT/hw1/hw_1_statement.tex`
- a part label such as `part b`
- any partial work already written by the student
- an optional request for a proof sketch, full proof, or LaTeX version

### Workflow

1. Look for a companion homework statement `.tex` file in the same folder and use it if present.
2. If no companion statement file exists, transcribe the homework PDF into one once, save it, and reuse it in later turns.
3. Read the relevant problem statement.
4. List the definitions, assumptions, and lemmas that seem relevant.
5. Propose a proof strategy before writing the full derivation.
6. Write a proof sketch with explicit logical steps.
7. Convert the sketch into a polished solution if requested.
8. Finish with a short self-check:
   - did the proof use every needed assumption?
   - did it prove the exact claim asked in the problem?
   - are there hidden gaps or unjustified jumps?

### Output Format

Unless the user asks for something else, respond in this order:

1. `Problem restatement`
2. `Key facts / assumptions`
3. `Proof idea`
4. `Proof sketch`
5. `Polished solution`
6. `Sanity check`
7. In LaTeX form, organize the polished solution with short subsection-style headings rather than one uninterrupted block.

### Style Rules

- Do not pretend a proof is complete if it still has gaps.
- Mark uncertainty explicitly.
- Keep notation consistent with the homework statement.
- Avoid introducing advanced machinery unless it clearly simplifies the argument.
- If the student already solved part of the problem, build on it instead of restarting from scratch.

### Example Prompts

- `/lt-proof Read AI4LT/hw1/hw_1.pdf and help me solve part b.`
- `/lt-proof Here is my draft for part b. Find the missing steps and rewrite it clearly.`
- `/lt-proof Turn my scratch work into a LaTeX-ready proof.`

## Sample Skill: `lt-latex-homework-format`

### Purpose

Turn a proof or solution into homework-ready LaTeX that matches the local course template and the user's current style preferences.

### Use When

Use this skill when the user asks for any of the following:

- write the final proof in LaTeX
- format a solution using the homework template
- revise notation, title, section layout, or equation style
- keep a stable course-specific writing style across assignments

### Inputs

Expect some or all of the following:

- source template: `AI4LT/assets/tex/hw_template.tex`
- solution content in plain text, sketch form, or polished form
- homework metadata such as assignment number, title, author, and section labels
- style overrides from the user

### Workflow

1. Read `AI4LT/assets/tex/hw_template.tex` before drafting the final LaTeX.
2. Preserve the existing preamble unless the user asks for a package or layout change.
3. Fill in title, author, and section headers using the current style preferences below.
4. Convert the proof into concise mathematical prose with consistent notation.
5. Use the preferred display-math style from the current style preferences.
6. Return either:
   - a drop-in LaTeX snippet for one problem, or
   - a full document fragment that fits directly into the template
7. If a user gives a new style rule, update this skill description first and then follow it.

### Output Format

Unless the user asks for something else, respond with:

1. `LaTeX snippet`
2. `Notes on filled fields or assumptions`

### Current Style Preferences

Treat this block as editable course memory. Update it when the user gives new formatting preferences.

- Base template: `AI4LT/assets/tex/hw_template.tex`
- Default deliverable: LaTeX
- Title style: use the assignment title in the template and adjust it when the user specifies a new title
- Problem structure: use assignment-aligned headings such as `\section{Part B: Theory Problem}` and `\subsection{Problem 1}` when they match the problem set
- Internal proof structure: break polished proofs into `\subsubsection{...}` blocks such as setup, case analysis, mistake bound derivation, and conclusion when that improves readability
- Case formatting: write case splits with bold labels such as `\textbf{Case 1: }`
- Display math: prefer `$$ ... $$` for unnumbered displayed equations
- Numbered equations: use `\begin{equation} ... \end{equation}` only when numbering is explicitly useful or requested
- Tone: concise, proof-oriented, and homework-ready

### Style Rules

- Keep the final LaTeX easy to paste into the template.
- Do not add fancy formatting unless the user asks for it.
- Keep notation aligned with the original problem statement.
- Prefer a visibly structured layout over one long proof block.
- If a proof is incomplete, reflect that honestly in the LaTeX instead of hiding the gap.
- When the user changes a style preference, treat the new preference as the latest default for this repo.

### Example Prompts

- `/lt-latex-homework-format Turn my solution for hw1 part b into LaTeX using the template.`
- `/lt-latex-homework-format Update the title to "Homework 1" and rewrite the display equations using $$ blocks.`
- `[lt-latex-homework-format] Format this proof as a section I can paste into the assignment file.`

## Template For New Skills

Copy this block and edit it when you want to add another skill:

```md
## Sample Skill: `skill-name`

### Purpose

Describe the one main job of the skill.

### Use When

- trigger 1
- trigger 2

### Inputs

- file or artifact
- user draft or notes

### Workflow

1. inspect the source material
2. plan the work
3. produce the requested output
4. check for mistakes

### Output Format

1. section one
2. section two

### Style Rules

- rule 1
- rule 2
```
