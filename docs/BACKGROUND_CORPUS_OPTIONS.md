# Background corpus options for H3 (S18)

H3 says augmentation with realistic backgrounds recovers part of the field gap.
S19 cannot start until a corpus exists that is licence-clean and **disjoint from
PlantDoc**. Using PlantDoc backgrounds as augmentation material would put the
field evaluation into the training set and invalidate the −78.4 pp result, which
is the project's main claim.

This file records what was checked and what remains unverified. It is written so
the decision can be made without redoing the search.

## Requirements a corpus has to meet

1. A licence that permits research use and redistribution of derived images.
2. Photographic field context: soil, grass, leaf litter, sky, hands, tools.
3. No overlap with PlantDoc, checked by hash rather than assumed.
4. Enough variety that the model cannot learn "background equals class".
5. Documented provenance, so the augmentation is reproducible.

## Candidate 1: DeepWeeds (recommended)

**What it is.** 17,509 in-situ images of eight weed species in rangeland pasture
across northern Queensland, published in *Scientific Reports* in 2019
(doi:10.1038/s41598-018-38343-3). Images were captured with a handheld camera in
the field, which is the exact character of imagery that PlantVillage lacks.

**Licence.** CC BY 4.0, stated in the repository README, on the Weed-AI dataset
page, and in the publisher's record. Attribution is required.

**Why it fits.** Different continent, different species, different cameras and a
different capture protocol from both PlantVillage and PlantDoc. Australia has no
overlap with PlantDoc's Indian field imagery in practice, and the species are
rangeland weeds rather than the 38 crop classes.

**The catch, stated plainly.** DeepWeeds images are framed on a weed, so the
"background" in each frame contains another plant. Using them wholesale as
backgrounds means the composite leaf sits among other vegetation. That is
arguably realistic, and it is what background augmentation does in the
literature, but it is not a clean soil plate. Two mitigations: sample only the
lower third of each frame where the ground dominates, or accept the vegetation
and say so.

**Unverified.** I have not downloaded the corpus or confirmed its internal file
layout, so I cannot state the exact number of usable background crops.

## Candidate 2: Poly Haven "farm soil" (fallback, weak)

**Licence.** CC0 1.0, so no attribution burden at all.

**The catch.** It is a 3D-scanned soil texture, not field photography. Compositing
leaves onto a uniform soil plate would produce backgrounds with none of the
clutter, occlusion or lighting variation that the field gap is made of. It would
almost certainly make Augmentation look effective for the wrong reason: the model
would learn to ignore a background that is trivially uniform.

Listed so the option is on record, not recommended.

## Candidate 3: general scene datasets (COCO, Open Images)

**Licence.** Mixed and per-image. COCO images are Flickr photographs under
various Creative Commons terms; Open Images is CC BY 4.0 for annotations with
image licences held by the original uploaders.

**The catch.** Filtering to a clean licence subset is real work, and the result is
mostly street scenes and interiors, which are not field context. Not worth it.

## What has to happen before S19

1. Download DeepWeeds and record its version and checksum.
2. Run `scripts/check_background_disjointness.py` against the PlantDoc manifests.
   It compares SHA-256 values, so overlap is detected rather than hoped against.
   Exact-hash comparison will not catch re-encoded near-duplicates; that
   limitation is stated in the script's output.
3. Decide the crop policy (full frame or ground-region sampling) and write it
   down before training, not after seeing results.
4. Then S19: retrain the reference model with background compositing, three
   seeds, everything else frozen, paired bootstrap on the PlantVillage to
   PlantDoc delta.

Until step 1 happens, H3 has no result and the abstract should not claim one.

## Attribution text, if DeepWeeds is used

> Background imagery from the DeepWeeds dataset (Olsen et al., *Scientific
> Reports* 9:2058, 2019), licensed CC BY 4.0.
