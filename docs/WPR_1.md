# Weekly Progress Report 1

**Project ID:** BT40
**Project title:** Plant Leaf Disease Detection Using CNN
**Team:** Shubham Kumar, Ujjwal Pratap Singh, Vivek Kumar Prajapati, Shubham Kumar
**Guide:** Dr. Harendra Singh
**Week commencing:** [fill in]

Two members of the team are named Shubham Kumar. Both are listed, in the order
given above, and enrolment numbers distinguish them.

## Work done this week

The project was audited before anything was changed. The repository held three
trained models but no reproducible evaluation path, and the EfficientNet-B0
checkpoint that most of the earlier results depended on was missing. I restored
it, then verified it structurally rather than trusting the filename: it loads with
`strict=True` into a 38-class model, has 4,056,226 parameters, and its class list
matches the development manifest in exact order.

I then ran the locked test evaluation and reproduced the recorded validation
macro-F1 of 0.99369 bit-identically. That number is now backed by a file with a
checksum instead of a claim in a document.

Two pieces of infrastructure were built. A single training engine replaces the
per-model scripts, so all architectures share one auditable code path. A complete
PlantDoc label map resolves all 28 field classes to PlantVillage classes, with
three rows marked as caveated and a written reason for each, one of which flags
that a class has only two training images and cannot support a per-class claim.

## Findings

The earlier project notes claimed a test accuracy of 99.51%. On re-measurement
that is correct: 10,709 test images, 53 errors. What the notes did not say is that
the test split has now been opened once, which is a fact any later reader needs.

The PlantDoc mapping in the repository covered 16 of 28 classes and marked every
row `review_required`, so no field evaluation was possible. That is now resolved.

## Problems

The first training engine run dropped the solution to a near-random model twice
with no error, because a learning-rate schedule decayed to 1% of its nominal value
by the last epoch. The schedule is now recorded per epoch in the training history,
so this cannot be invisible again.

## Next week

Freeze one training protocol, run a pilot across all four architectures to check
memory and throughput, and prepare the seeded comparison sweep.
