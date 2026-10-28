"""Do autistic children fall into subtypes based on the other conditions they have?

This package holds the reusable pieces of that analysis. The numbered scripts
in ``scripts/`` string them together; see the README for the overall picture
and ``docs/background.md`` if the data or the methods are new to you.

Modules, in the order the pipeline uses them:

- ``config``: reads ``config/conditions.yaml``, which lists the survey questions we use.
- ``data``: downloads the Census files, selects autistic children aged 6 to 17,
  and turns their answers into 0/1 condition flags (steps 1 and 2).
- ``lca``: the main clustering method, latent class analysis (step 3).
- ``hierarchical`` and ``partition``: two alternative methods used as a cross-check (step 4).
- ``reference``: library versions of all three methods, for comparison and as a fallback.
- ``backends``: picks "ours" or "reference" by name.
- ``synthetic``: fake data with known classes, for testing.
- ``validate``: the checks the clusters have to pass (steps 4 to 14), including the
  severity test in step 7 and the pairwise-fit check in step 12.
- ``describe``: national shares and class profiles using survey weights (step 7).
- ``latent_trait``: the continuous alternative to classes, a model with one to four
  dimensions instead of K groups (steps 12 and 13).
- ``plots``: shared figure styling.
"""

__version__ = "0.1.0"
