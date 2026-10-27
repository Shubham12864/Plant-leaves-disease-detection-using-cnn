#!/usr/bin/env bash
# Fails if the university submission document (thesis/submission) or the form
# answers still contain internal notes, placeholders, or statements the
# repository evidence contradicts. Every pattern below was found in a draft.
set -u
cd "$(dirname "$0")/.."

status=0
check() {  # check <description> <regex> <paths...>
  local what="$1" pattern="$2"; shift 2
  local hits
  hits=$(grep -rnE --include='*.tex' --include='*.md' "$pattern" "$@" 2>/dev/null || true)
  if [ -n "$hits" ]; then
    echo "FAIL: $what"
    echo "$hits" | sed 's/^/    /' | cut -c1-170
    status=1
  fi
}

SUB=thesis/submission
FORM=docs/SUBMISSION_FORM_ANSWERS.md

# Internal notes that must not ship in the submitted PDF
check "red placeholder left in the document"            '\\placeholder'                          $SUB
check "internal 'verify before submitting' box"         'Verify before submitting'               $SUB
check "internal pre-submission checklist"               'Pre-submission checklist'               $SUB
check "coordinator deadline note on the cover"          "coordinator.s mail: 29"                 $SUB

# Statements contradicted by the repository
check "test split said to be opened once (D11: scored more than once)"  'opened once'           $SUB
check "sweep duration with no recorded source (R13 is unexplained)"     'about 2\.6 hours|2\.6[ ~]*hours' $SUB
check "test-catch claim with no supporting test"        'metrics function signature'             $SUB
check "intervention-experiment output-contract test does not exist"     "intervention experiment.s output" $SUB
check "0.1737 is the published checkpoint, not a five-seed mean"        'cohort A, averaged over five seeds' $SUB
check "81-point drop belongs to 0.1737; the five-seed mean drop is 78 points" '81-point drop'    $SUB
check "H4 reworded away from the registered hypothesis" 'Confirmed as a negative finding'        $SUB
check "mid-term and Phase III disagree on the label mapping" 'described the mapping as partially complete' $SUB
check "PlantVillage total does not equal the sum of its splits (54,015)" '54,305'                $SUB $FORM
check "dissertation PDF is not in the repository, and it is a draft"    'BT40_dissertation_draft|47-page' $SUB

# Every GitHub link pasted into the form must point at a tracked file, or the evaluator gets a 404
for p in $(grep -oE '(blob|tree)/main/[^ )]+' "$FORM" | sed -E 's#^(blob|tree)/main/##' | sort -u); do
  if [ -z "$(git ls-files -- "$p")" ]; then
    echo "FAIL: form links to a path git does not track (404 on GitHub): $p"
    status=1
  fi
done

# Long file paths must be breakable, or they run off the page
if [ -f thesis/submission/build/main.log ]; then
  over=$(grep -c 'Overfull .hbox' thesis/submission/build/main.log || true)
  if [ "${over:-0}" -gt 0 ]; then
    echo "FAIL: $over overfull lines in the built PDF (text runs past the margin); rebuild after fixing"
    status=1
  fi
fi

[ "$status" -eq 0 ] && echo "submission documents are consistent with the repository evidence"
exit "$status"
