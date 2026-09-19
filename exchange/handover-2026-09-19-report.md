# Handover report 2026-09-19 (opus pane)

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-19-handover-opus.md` plus the addendum (part D).

## A. Android release

**A1 DONE.** Clean clone of `main` (`808ea9f`) at `~/Android/.work/release-clone`.
`tools/fetch-llama.sh` printed `OK llama.cpp ca3d5a3e1 in third_party/llama.cpp`, so tonight's
fetch step and the Gradle lookup change are now verified, not assumed.
`./gradlew --no-daemon assembleRelease testDebugUnitTest` ended `BUILD SUCCESSFUL`, exit 0, 199
tasks, `:app:testDebugUnitTest` among them. The keystore passwords came from the desktop keyring
through the environment of that one command and were never printed, logged or committed.

**A2 DONE.** `apksigner verify --print-certs` (build-tools 36.0.0) reports Signer #1 `CN=baro`,
cert sha256 `b87b7d5f...bc5b`. Receipt committed as `exchange/release-0.1.0+808ea9f.txt` in
`~/Android/baro` (`3d8cc71`): APK sha256
`b24cb8998f3a8418aa5ecfa38d8f1c4799b0f548082a0978b4c2467bcb49af04`, 70222972 bytes, versionName
`0.1.0+808ea9f`, versionCode 82. The APK itself is at `~/Android/releases/baro-0.1.0+808ea9f.apk`,
outside git. Export re-run after the commit, `PASS -- all verifications clean`, only `main` left.

**A3 DONE (the line is in "For the maintainer" below).**

## B. lane MOEPF

**BLOCKED, not started.** the maintainer said "no more runs, I need the system" at 22:38 and has not freed
the card. No GPU job was queued. Items B1 to B6 are untouched; the `moepf` worktree and branch
still exist.

One B item that is CPU only was done: `tools/ci-checks.sh` ran green on `main`
(`all non-GPU checks passed`, exit 0) after tonight's commits, and `bench/preflight.sh` was
re-stamped (`PASS preflight ddfae0dffb93`, `--check` exit 0). The `BARO_PACK=/nonexistent` skip
branch of `run-tests.sh` is still UNVERIFIED: it needs a run under `gpu-wait`.

## C. Publish

**DONE for both repos, nothing pushed.**
`~/Projects/mojo/mojo-baro-public-20260919` (tip = the ci-checks fix) and
`~/Android/baro-public-20260919` (tip = the release receipt), each `PASS -- all verifications
clean`, gitleaks clean, one branch `main`, no remote. Re-export after any further public commit.

The README paragraph about `bench/` and `exchange/` and the Hugging Face GGUF line were NOT
written: both were marked ask-first and the maintainer has not answered.

## D. Addendum (skills and iTools)

1. **DONE.** `publish-purge` gained `--keep-branch NAME` (checks it out in OUT, deletes every
   other local branch, fails loudly on an unknown name with the branches left alone) and a
   "largest blobs in the rewritten history (top 10)" section with the pack size.
   `test_publish_purge.py` covers both: 25 checks, ends `PASS`. iTools `dad79ea`, `840136b`;
   `index-gen --write` re-run and the row carries the new usage line.
   The first version of the blobs report killed the run: `head -10` closing the pipe under
   `set -o pipefail` aborted the script before gitleaks, `--keep-branch` and the verdict, and a
   real export came out with all eighteen lane branches in it. Fixed in `6b71f08` (subshell with
   pipefail off) and covered by a fixture with more blobs than the report lists, so the truncating
   case is actually exercised: 27 checks, PASS. Both exports were then rebuilt with
   `--keep-branch main`, each ending `PASS -- all verifications clean` with one branch.
2. **DONE.** New skill `~/Brain/Skills/clean-publish/SKILL.md`, indexed in `Skills/_index.md`,
   symlinked into `~/.claude/skills/.user/clean-publish` by `skill-new.sh`. Brain `62623db5`.
3. **DONE.** `herdr-ops`: `--effort` marked REQUIRED next to the explicit model in the verified
   start example, with the `w9C:p4` observation as the evidence.
4. **DONE.** `tools/ci-checks.sh` writes its KERNELS.md temp copy to `.work/ci-kernels-before.md`
   (mojo-baro `6a876b5`). Check: `tools/ci-checks.sh` exit 0, `git status --short` empty after the
   run, `bench/preflight.sh --check` exit 0.
5. **DONE.** `gate-authoring` rule 4: nothing writes into the tree while preflight checks are still
   ahead in the chain; docs edits go in a separate worktree on the same branch.

## For the maintainer

Publish the APK as a release asset once `github.com/amarbaro/baro.apk` exists:

```
gh release create v0.1.0 ~/Android/releases/baro-0.1.0+808ea9f.apk --repo amarbaro/baro.apk --title "baro 0.1.0" --notes "arm64-v8a + x86_64, minSdk 31, sha256 b24cb8998f3a8418aa5ecfa38d8f1c4799b0f548082a0978b4c2467bcb49af04"
```

Push, from the export folders, nobody else does this:

```
cd ~/Projects/mojo/mojo-baro-public-20260919 && git remote add origin https://github.com/amarbaro/mojo-baro.git && git push -u origin main
cd ~/Android/baro-public-20260919 && git remote add origin https://github.com/amarbaro/baro.apk.git && git push -u origin main
```

Open questions unchanged from the baton: the MoE `BARO_PREFILL` default flip, the two unconfirmed
`CLAUDE.md` rules, the Hugging Face repo id for the README, and whether the `bench/`+`exchange/`
paragraph goes in.
