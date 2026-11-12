# HardPair

Track B code for the HardPair team project: pathology image classification,
regional explanations, and feedback-based classifier updates.

## Approach

The workflow combines three components:

- Classification: a linear probe combines CONCH and MUSK image representations
  to predict tissue classes.
- Regional evidence: image regions associated with the predicted class and its
  closest alternative guide explanations from a LoRA-adapted Qwen2-VL-2B.
- Targeted feedback: encoder disagreement, prediction margins, and feature
  diversity select examples for additional labels, with extra coverage for
  muscle–stroma confusion.

The classifier and explanation model serve different roles. Feedback updates
the classifier, while the explanation model connects its predictions to
regional evidence for review.

## Results

On 3,590 evaluation tiles, increasing simulated feedback from 200 to 400 labels
raised balanced accuracy from 97.96% to 99.35%. Muscle–stroma mutual error fell
from 5.52% to 1.18%.

The classifier also used 16,200 training tiles. Feedback and evaluation came
from separate splits of the CRC-VAL-HE-7K domain. Regional explanations passed
the recorded format and grounding checks for 54.6% of evaluation tiles.

## Review workflow

The Streamlit application brings predictions, regional evidence, and feedback
queries into one interface. Reviewers can inspect a prediction alongside its
alternative class and trace how selected labels enter the next classifier
update.

## Documentation

- [Application and implementation guide](trackb/README.md)
- [Evaluation and feedback results](trackb/docs/validation/gpu_final_2026-10-18/README.md)
- [Upstream project](https://github.com/alin4638/hardpair-trackb-handoff)
