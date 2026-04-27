"""
llm_filter

Responsibilities:
- Provide "knowledge content filtering": classify PDF-extracted paragraphs as KEEP/DROP.
- Goal: remove low-knowledge-density content such as introduction/background, patient case
  narratives, future outlook, and methodological limitations, while retaining abstracts,
  results, guidelines/conclusions, and table information (tables are always kept).
"""


