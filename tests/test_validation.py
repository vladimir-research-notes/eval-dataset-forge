"""Tests for the validation & redaction layer (Milestone 3).

Covered here:

  * **PII/secret detection** — positive hits for each detector and, importantly,
    negatives on code / dates / small numbers so the checks are not noise.
  * **Redaction** — deterministic replacement, index-safe multi-span redaction,
    key-name preservation for assignments, and the guarantee that a report
    snippet never contains the raw matched value.
  * **Schema validation** — every SchemaError kind (parse, missing, unknown,
    type), lenient multi-error reporting, and duplicate-id detection.
  * **Quality warnings** — empty input / empty expected_output / unknown family.
  * **Whole-dataset validation** — the committed fixtures are clean (the
    safe-data guarantee), the severity policy behaves, and the CLI exit codes and
    ``--redact-out`` round-trip work.

Everything is CPU-only and offline. All "secrets" here are synthetic (the public
AWS documentation example key, obviously fake tokens).
"""

from __future__ import annotations

import json

import pytest

from eval_dataset_forge.datasets import loader
from eval_dataset_forge.schemas import DatasetExample
from eval_dataset_forge.validation import (
    ValidationReport,
    format_report,
    validate_dataset,
    write_redacted,
)
from eval_dataset_forge.validation.pii import (
    ALL_KINDS,
    PII_KINDS,
    SECRET_KINDS,
    scan,
    scan_example,
    shannon_entropy,
)
from eval_dataset_forge.validation.quality import check_quality
from eval_dataset_forge.validation.redaction import (
    PLACEHOLDERS,
    redact,
    redact_example,
    redacted_context,
)
from eval_dataset_forge.validation.schema import (
    find_duplicate_ids,
    load_records_lenient,
    validate_records,
)

# A synthetic, obviously fake AWS access-key id (the AWS docs example value).
FAKE_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
FAKE_TOKEN = "sk-A1b2C3d4E5f6G7h8I9j0KLMN"


# --------------------------------------------------------------------------- #
# PII / secret detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, kind",
    [
        ("reach me at alice.smith@example.com please", "email"),
        (f"leaked key {FAKE_AWS_KEY} in logs", "aws_access_key"),
        ("data -----BEGIN RSA PRIVATE KEY----- more", "private_key"),
        ("config api_key = s3cretValue12", "secret_assignment"),
        (f"authorization bearer {FAKE_TOKEN} end", "token"),
        ("server 10.0.100.25 replied", "ipv4"),
        ("ring +1 (415) 555-0142 now", "phone"),
    ],
)
def test_scan_detects_each_kind(text: str, kind: str) -> None:
    spans = scan(text)
    assert kind in {s.kind for s in spans}, spans


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The capital of France is Paris.",
        "def reverse(s):\n    return s[::-1]",
        "for i in range(1, n + 1): if i % 15 == 0",
        "success_criteria: reverse('abc') == 'cba'",
        "created on 2024-01-03 at build 4471",
        "password: required",  # prose, not a real secret value
        "abcdefghijklmnopqrstuvwxyz",  # long but single-class, low entropy
    ],
)
def test_scan_negatives(text: str) -> None:
    assert scan(text) == []


def test_scan_is_deterministic() -> None:
    text = f"key {FAKE_AWS_KEY} and mail bob@example.org"
    assert scan(text) == scan(text)


def test_scan_spans_do_not_overlap() -> None:
    spans = scan(f"{FAKE_AWS_KEY} then mail a@b.co then 10.1.2.3")
    ordered = sorted(spans, key=lambda s: s.start)
    for prev, nxt in zip(ordered, ordered[1:], strict=False):
        assert prev.end <= nxt.start


def test_ipv4_range_validation() -> None:
    # 999.1.1.1 is not a valid dotted quad and must not match.
    assert not any(s.kind == "ipv4" for s in scan("visit 999.1.1.1 today"))


def test_shannon_entropy_bounds() -> None:
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_secret_and_pii_kind_partition() -> None:
    assert SECRET_KINDS.isdisjoint(PII_KINDS)
    assert ALL_KINDS == SECRET_KINDS | PII_KINDS


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_redact_replaces_with_typed_placeholder() -> None:
    assert redact("mail me at x@y.com") == "mail me at <EMAIL>"
    assert redact(f"key {FAKE_AWS_KEY}!") == "key <AWS_KEY>!"


def test_redact_is_deterministic_and_idempotent_on_clean_text() -> None:
    clean = "What is the capital of France?"
    assert redact(clean) == clean
    once = redact("a@b.com and 10.0.0.1")
    assert redact(once) == once  # placeholders contain no PII to re-redact


def test_redact_multiple_spans_index_safe() -> None:
    text = f"m1 a@b.com m2 {FAKE_AWS_KEY} m3 10.0.0.1 m4"
    assert redact(text) == "m1 <EMAIL> m2 <AWS_KEY> m3 <IP> m4"


def test_secret_assignment_keeps_key_name() -> None:
    # The key name stays; only the value is redacted.
    assert redact("api_key = s3cretValue12") == "api_key = <SECRET>"


def test_placeholders_cover_all_kinds() -> None:
    assert ALL_KINDS <= set(PLACEHOLDERS)


def test_redacted_context_never_leaks_raw_value() -> None:
    text = f"the secret token is {FAKE_TOKEN} keep it safe"
    span = next(s for s in scan(text) if s.kind == "token")
    snippet = redacted_context(text, span)
    assert FAKE_TOKEN not in snippet
    assert "<TOKEN>" in snippet


def test_redact_example_scrubs_nested_fields() -> None:
    ex = DatasetExample(
        example_id="e1",
        source_id="s1",
        task_family="qa",
        input="write to admin@corp.example",
        expected_output="ok",
        metadata={"note": f"deploy key {FAKE_AWS_KEY}", "docs": ["ip 10.9.8.7"]},
    )
    red = redact_example(ex)
    assert red.input == "write to <EMAIL>"
    assert red.metadata["note"] == "deploy key <AWS_KEY>"
    assert red.metadata["docs"] == ["ip <IP>"]
    # Original is untouched.
    assert FAKE_AWS_KEY in ex.metadata["note"]


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def _indexed(records: list[dict]):
    return list(enumerate(records, start=1))


def test_validate_records_accepts_valid_record() -> None:
    rec = {
        "example_id": "e1",
        "source_id": "s1",
        "task_family": "qa",
        "input": "hi",
        "expected_output": "a",
    }
    examples, errors = validate_records(_indexed([rec]))
    assert not errors
    assert [ex.example_id for _, ex in examples] == ["e1"]


def test_validate_records_flags_missing_and_unknown_and_type() -> None:
    records = [
        {"source_id": "s", "task_family": "qa", "input": "x", "expected_output": "a"},  # missing id
        {
            "example_id": "e2",
            "source_id": "s",
            "task_family": "qa",
            "input": "x",
            "expected_output": "a",
            "oops": 1,  # unknown field
        },
        {
            "example_id": "e3",
            "source_id": "s",
            "task_family": "qa",
            "input": 123,  # type error: input must be str
            "expected_output": "a",
        },
    ]
    _, errors = validate_records(_indexed(records))
    kinds = {e.kind for e in errors}
    assert "missing_field" in kinds
    assert "unknown_field" in kinds
    assert "type_error" in kinds


def test_lenient_reader_reports_bad_json_line(tmp_path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"example_id":"e1","source_id":"s","task_family":"qa","input":"x","expected_output":"a"}\n'
        "{not json}\n",
        encoding="utf-8",
    )
    indexed, parse_errors = load_records_lenient(path)
    assert len(indexed) == 1  # the good line survives
    assert [e.kind for e in parse_errors] == ["parse_error"]


def test_find_duplicate_ids() -> None:
    records = [
        {"example_id": "dup"},
        {"example_id": "unique"},
        {"example_id": "dup"},
    ]
    dupes = find_duplicate_ids(_indexed(records))
    assert dupes == {"dup": [1, 3]}


# --------------------------------------------------------------------------- #
# Quality warnings
# --------------------------------------------------------------------------- #


def test_quality_warnings() -> None:
    examples = [
        DatasetExample(
            example_id="q1", source_id="s", task_family="qa", input="  ", expected_output="a"
        ),
        DatasetExample(
            example_id="q2", source_id="s", task_family="qa", input="ok", expected_output=""
        ),
        DatasetExample(
            example_id="q3", source_id="s", task_family="mystery", input="ok", expected_output="a"
        ),
    ]
    warnings = {(w.example_id, w.kind) for w in check_quality(examples)}
    assert ("q1", "empty_input") in warnings
    assert ("q2", "empty_expected_output") in warnings
    assert ("q3", "unknown_task_family") in warnings


# --------------------------------------------------------------------------- #
# Whole-dataset validation over the committed fixtures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture_dir", loader.discover_fixtures(), ids=lambda p: p.name)
def test_fixtures_have_no_errors(fixture_dir) -> None:
    """Every committed fixture is schema-valid, uniquely-keyed, and PII-free.

    This is the safe-data guarantee: the audit can only claim the fixtures are
    public-safe if scanning them yields no PII/secret findings and no structural
    errors. Empty gold answers are planted defects and surface only as warnings.
    """
    report = validate_dataset(fixture_dir / "examples.jsonl")
    assert report.schema_errors == []
    assert report.duplicate_ids == {}
    assert report.pii_findings == []
    assert report.ok


def test_fixture_missing_label_is_a_warning_not_error() -> None:
    report = validate_dataset(loader.FIXTURES_ROOT / "qa_toy" / "examples.jsonl")
    assert report.ok  # warnings do not fail validation
    kinds = {(w.example_id, w.kind) for w in report.quality_warnings}
    assert ("qa-010", "empty_expected_output") in kinds


# --------------------------------------------------------------------------- #
# Severity policy + reporting
# --------------------------------------------------------------------------- #


def _write(tmp_path, name: str, records: list[dict]):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_email_is_warning_but_promotable(tmp_path) -> None:
    path = _write(
        tmp_path,
        "mail.jsonl",
        [
            {
                "example_id": "e1",
                "source_id": "s",
                "task_family": "qa",
                "input": "write to a@b.com",
                "expected_output": "x",
            }
        ],
    )
    lenient = validate_dataset(path)
    assert lenient.ok  # email alone is a warning
    assert len(lenient.pii_findings) == 1

    strict = validate_dataset(path, fail_on_pii=True)
    assert not strict.ok  # promoted to an error


def test_secret_fails_even_without_fail_on_pii(tmp_path) -> None:
    path = _write(
        tmp_path,
        "secret.jsonl",
        [
            {
                "example_id": "e1",
                "source_id": "s",
                "task_family": "qa",
                "input": f"key {FAKE_AWS_KEY}",
                "expected_output": "x",
            }
        ],
    )
    report = validate_dataset(path)
    assert not report.ok
    assert report.counts()["secret_findings"] == 1


def test_report_to_dict_is_json_serializable(tmp_path) -> None:
    report = validate_dataset(loader.FIXTURES_ROOT / "qa_toy" / "examples.jsonl")
    dumped = json.dumps(report.to_dict())
    assert '"ok": true' in dumped
    assert "counts" in json.loads(dumped)


def test_format_report_contains_verdict() -> None:
    report = validate_dataset(loader.FIXTURES_ROOT / "qa_toy" / "examples.jsonl")
    text = format_report(report)
    assert "verdict" in text
    assert "OK" in text


# --------------------------------------------------------------------------- #
# CLI + redacted round-trip
# --------------------------------------------------------------------------- #


def test_cli_validate_exit_codes(tmp_path) -> None:
    from eval_dataset_forge.cli import main

    good = loader.FIXTURES_ROOT / "qa_toy" / "examples.jsonl"
    assert main(["validate", "--dataset", str(good)]) == 0

    dup = {"example_id": "d", "source_id": "s", "task_family": "qa", "expected_output": "a"}
    bad = _write(
        tmp_path,
        "bad.jsonl",
        [{**dup, "input": "x"}, {**dup, "input": "y"}],
    )
    assert main(["validate", "--dataset", str(bad)]) == 1  # duplicate id


def test_write_redacted_roundtrip(tmp_path) -> None:
    src = _write(
        tmp_path,
        "src.jsonl",
        [
            {
                "example_id": "e1",
                "source_id": "s",
                "task_family": "qa",
                "input": f"contact a@b.com key {FAKE_AWS_KEY}",
                "expected_output": "x",
            }
        ],
    )
    dst = tmp_path / "redacted.jsonl"
    n = write_redacted(src, dst)
    assert n == 1
    # The redacted output re-validates clean, with no PII.
    report = validate_dataset(dst)
    assert report.pii_findings == []
    assert "<EMAIL>" in dst.read_text(encoding="utf-8")
    assert FAKE_AWS_KEY not in dst.read_text(encoding="utf-8")


def test_scan_example_reports_field_path() -> None:
    ex = DatasetExample(
        example_id="e1",
        source_id="s",
        task_family="qa",
        input="hello",
        expected_output="ok",
        metadata={"retrieval": {"docs": ["reach me at a@b.com"]}},
    )
    findings = scan_example(ex)
    assert len(findings) == 1
    assert findings[0].field == "metadata.retrieval.docs[0]"
    assert findings[0].kind == "email"


def test_validation_report_dataclass_defaults() -> None:
    empty = ValidationReport(dataset_path="x", n_records=0, n_valid=0)
    assert empty.ok
    assert empty.counts()["errors"] == 0
