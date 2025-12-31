"""
Lightweight, dependency-free tests for post-processing.

Run:
  python external_kv_injection/scripts/test_postprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.runtime.postprocess import (  # type: ignore
        generic_text_hygiene,
        postprocess_answer,
        schema_aware_formatter,
    )
except ModuleNotFoundError:
    from src.runtime.postprocess import generic_text_hygiene, postprocess_answer, schema_aware_formatter  # type: ignore


class TestGenericTextHygiene(unittest.TestCase):
    def test_removes_prompt_echo_prefix(self):
        user_prompt = "What is X?"
        raw = "What is X?\n\nX is a thing.\n"
        out = generic_text_hygiene(raw, user_prompt=user_prompt)
        self.assertEqual(out, "X is a thing.")

    def test_dedupes_repeated_sentences_conservatively(self):
        raw = "A is B. A is B. A is B.\n\nC is D."
        out = generic_text_hygiene(raw, user_prompt=None)
        self.assertIn("A is B.", out)
        # Should not contain 3 repeats after hygiene.
        self.assertLessEqual(out.count("A is B."), 1)

    def test_drops_degenerate_symbol_lines(self):
        raw = "【证据句】】】】】】】】】】】\n\nOK."
        out = generic_text_hygiene(raw, user_prompt=None)
        self.assertEqual(out, "OK.")

    def test_removes_boilerplate_tail_paragraph(self):
        raw = "Answer content.\n\n如有其他问题，请随时告知。"
        out = generic_text_hygiene(raw, user_prompt=None)
        self.assertEqual(out, "Answer content.")


class TestSchemaAwareFormatter(unittest.TestCase):
    def test_no_answered_slots_returns_unchanged(self):
        s = "Plain answer."
        self.assertEqual(schema_aware_formatter(s, answered_slots=None, slot_to_section_title=None), s)
        self.assertEqual(schema_aware_formatter(s, answered_slots=[], slot_to_section_title={}), s)

    def test_filters_by_titles_from_mapping_only(self):
        cleaned = "Section A:\nalpha\n\nSection B:\nbeta\n"
        out = schema_aware_formatter(
            cleaned,
            answered_slots=["slot_a"],
            slot_to_section_title={"slot_a": "Section A", "slot_b": "Section B"},
        )
        self.assertIn("Section A：", out)
        self.assertIn("alpha", out)
        self.assertNotIn("beta", out)

    def test_if_no_headings_detected_returns_original(self):
        cleaned = "No headings here.\nJust text."
        out = schema_aware_formatter(
            cleaned,
            answered_slots=["slot_a"],
            slot_to_section_title={"slot_a": "Section A"},
        )
        self.assertEqual(out, cleaned.strip())


class TestWrapper(unittest.TestCase):
    def test_wrapper_orchestrates(self):
        raw = "Evidence:\nA is B.\nA is B."
        out = postprocess_answer(raw, user_prompt=None)
        self.assertLessEqual(out.count("A is B."), 1)


if __name__ == "__main__":
    unittest.main()


