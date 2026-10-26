# S16 — how to fill this in

You are labelling 50 heat maps. It takes about 15 minutes and it finishes H2.

## Why this exists

We already know the model breaks when the background changes. Removing the
background costs it 56.5 points on PlantVillage, so a lot of its 99% was never
about the leaf.

What we do not know is whether it is **looking** at the background or merely
**affected** by it. Those are different claims, and the report cannot say "the
model reads the soil" without evidence for the looking part. This is that
evidence.

## What you need open

| file | what it is |
|---|---|
| `contact_sheet.png` | 50 heat maps, numbered 1 to 50, in one image |
| `labelling_sheet.csv` | 50 matching rows, one per number |
| `tiles/01.png` … `tiles/50.png` | the same maps individually, if you want to zoom |

## The four steps

1. Open `contact_sheet.png` and zoom in. It is 2775 × 1612, so it stays sharp.
2. Open `labelling_sheet.csv` beside it.
3. For each number 1 to 50, look at where the **red and orange** is heaviest and
   type one of these three words into the `peak_region` column:

| word | when to use it |
|---|---|
| `leaf` | the hot spot sits on plant tissue, whether healthy, spotted or diseased |
| `background` | the hot spot sits mostly on soil, grass that is not the subject, sky, a hand, a tool, or printed text |
| `borderline` | genuinely split, roughly half and half. Don't force a call |

4. Save the CSV. That's it.

## How to decide quickly

Look at the brightest patch and ask: *if I covered that patch, would the model
lose the leaf or the surroundings?* Trust your first impression. A second opinion
would not agree with you on every tile anyway, and forcing agreement is not the
point.

Do not look at the `predicted_class` or `confidence` columns while labelling. They
are there for later analysis, and seeing them nudges you toward "the model was
right so it must have looked at the leaf", which is the assumption being tested.

Use `notes` only if something jumps out, for example "hot spot is on the printed
caption" or "leaf almost fills the frame".

## About the last column before you

`automated_mask_proxy` is a number I computed from a colour heuristic. It is not a
label and it is not a prediction of what you should write. **Ignore it while
labelling.** Its only purpose is to be compared against your answers afterwards:
if your labels and the heuristic agree, the heuristic can be trusted for the full
236-image set; if they disagree, your labels win and the heuristic gets reported
as unreliable. That comparison is itself a result.

## What happens when you send it back

I compute the rate, which is one sentence of the form "X of 50 field heat maps
peak on leaf tissue, Y on background". That sentence either supports H2 or
contradicts it, and both are publishable. Then it goes into
`reports/mechanism/MECHANISM_RESULTS.md`, the registry, and the results section,
and H2 stops being marked "partially supported".

## A note on what I already noticed

Two tiles (13 and 43) put their hot spot on **printed text inside the photo**.
Some PlantDoc images have captions baked into the frame. If that pattern shows up
more often, it is worth a sentence in the report: the model may be reading letters
rather than leaves. That is exactly the kind of thing a human notices and a
heuristic does not.
