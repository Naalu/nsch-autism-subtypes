"""Subtypes of autistic children in the National Survey of Children's Health.

The package is organized by pipeline step:

- ``config``: reads ``config/conditions.yaml``.
- ``data``: downloads the Census files, selects autistic children aged 6-17,
  and encodes the condition flags (steps 1-2).
- ``lca`` and ``hierarchical``: the two clustering methods, written from
  scratch for the course (step 3 and step 4).
- ``reference``: StepMix and SciPy wrappers used to check them.
- ``validate``: class matching, ARI, bootstrap stability, the count-vs-class
  test (steps 4-6 and 8).
- ``describe``: survey-weighted shares and class profiles (step 7).
"""

__version__ = "0.1.0"
