# Segmentation

**Status: planned.** The current application has no segmentation observation, mask-consuming task, or segmentation backend.

Segmentation partitions image pixels into regions. The first documented subtopic is **semantic segmentation**, which assigns a category per pixel and does not distinguish separate instances of the same category. Instance segmentation separates object instances; panoptic approaches combine semantic and instance distinctions. These should remain separate capability choices because they have different contracts and evaluation needs.

Before choosing a model, name the consumer (for example, table/workspace masking or object-boundary refinement), required classes and resolution, coordinate/resize mapping, confidence representation, throughput target, and memory budget. Evaluate per-class intersection-over-union and boundary quality on held-out labeled frames, then measure whether the downstream task improves. Split this page if additional segmentation modes become concrete enough to warrant independent contracts and evaluation.
