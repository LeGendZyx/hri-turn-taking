"""Human-facing, non-destructive Transcript review UI and audio clips.

The UI never edits corpus audio or a prior draft.  It renders short temporary
WAV clips from the correction-mapped read-only A/B/C/R sources and saves each
checkpoint as a new, fully replayed draft revision.  Playback, navigation and
automatic display are never treated as evidence of human review; only the
explicit ``Apply reviewed decision`` action sets ``reviewed=true``.
"""

from __future__ import annotations

import array
import copy
import os
import stat
import sys
import tempfile
import wave
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from hri_turn_taking.data import audio_contract as audio_contract_module
from hri_turn_taking.data import transcript_acceptance as acceptance_module
from hri_turn_taking.data import transcript_fs_guard as fs_guard
from hri_turn_taking.data import transcript_review as review_module

_CLOUD_RECALL_FLAGS = (
    getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_OPEN", 0x40000)
    | getattr(stat, "FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS", 0x400000)
)


class TranscriptReviewUIError(ValueError):
    """Raised when review audio or a draft checkpoint is unsafe."""


@dataclass(frozen=True)
class ReviewWorkspace:
    """A validated editable draft and its current revision path."""

    path: Path
    draft: dict[str, Any]


def _fail(message: str) -> None:
    raise TranscriptReviewUIError(message)


def _file_attributes(path: Path) -> int:
    try:
        return int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError as exc:
        raise TranscriptReviewUIError(f"cannot inspect review audio: {path}") from exc


def _require_online_regular_file(path: Path, label: str) -> Path:
    try:
        guarded = fs_guard.guard_regular_file_path(path)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewUIError(f"{label} path is unsafe: {exc}") from exc
    attributes = _file_attributes(guarded)
    if attributes & _CLOUD_RECALL_FLAGS:
        _fail(
            f"{label} is a OneDrive Offline placeholder; choose 'Always keep on this "
            "device' in Explorer before review. The tool will not hydrate it implicitly."
        )
    return guarded


def _preflight_audio(
    scene: str, authority: review_module.ReviewSourceAuthority
) -> None:
    review_module._scene_parts(scene)
    audio_authority = audio_contract_module.AudioSourceAuthority(
        corpus_root=authority.corpus_root,
        hpe_delivery_root=authority.hpe_delivery_root,
    )
    paths = audio_contract_module.canonical_source_paths(scene, audio_authority)
    for speaker in review_module.SPEAKERS:
        _require_online_regular_file(paths[speaker], f"{speaker} lavalier WAV")
    _require_online_regular_file(paths["master"], "aligned four-channel master WAV")
    _require_online_regular_file(paths["reference"], "Transcript reference WAV")


def preflight_review_audio(scene: str) -> None:
    """Fail before source replay if a production review WAV is not local."""

    _preflight_audio(scene, review_module.default_source_authority())


def _scene_from_untrusted_draft_bytes(raw: bytes) -> str:
    value = review_module._require_object(
        review_module._strict_json_loads(raw, "review draft preflight"),
        "review draft preflight",
    )
    scene = review_module._require_string(value.get("scene"), "scene")
    review_module._scene_parts(scene)
    return scene


def _load_workspace_with_authority(
    path: str | Path,
    *,
    authority: review_module.ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
    formal_snapshot: review_module.CanonicalDraftSnapshot | None = None,
) -> ReviewWorkspace:
    try:
        draft_path = fs_guard.guard_regular_file_path(path)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewUIError(f"draft path is unsafe: {exc}") from exc
    with fs_guard.guarded_read(draft_path) as held_draft:
        raw = fs_guard.read_guarded_bytes(held_draft)
        if formal_snapshot is not None and (
            draft_path != formal_snapshot.path or raw != formal_snapshot.raw
        ):
            _fail("actual UI draft differs from the held canonical snapshot")
        scene = _scene_from_untrusted_draft_bytes(raw)
        # This check runs before any full source replay so a cloud placeholder is
        # never opened as a side effect of starting the UI.
        _preflight_audio(scene, authority)
        try:
            draft = review_module._load_review_draft_with_authority(
                draft_path,
                authority=authority,
                plan=plan,
                formal_scene=formal_scene,
                formal_snapshot=formal_snapshot,
            )
        except (review_module.TranscriptReviewError, OSError) as exc:
            raise TranscriptReviewUIError(f"draft replay failed: {exc}") from exc
        fs_guard.verify_guarded_read(held_draft)
    return ReviewWorkspace(draft_path, copy.deepcopy(draft))


def load_review_workspace(path: str | Path) -> ReviewWorkspace:
    """Load a production draft after offline-placeholder preflight."""

    try:
        with review_module.held_canonical_draft(path) as snapshot:
            return _load_workspace_with_authority(
                snapshot.path,
                authority=review_module.default_source_authority(),
                formal_scene=snapshot.scene,
                formal_snapshot=snapshot,
            )
    except review_module.TranscriptReviewError as exc:
        raise TranscriptReviewUIError(f"formal draft path gate failed: {exc}") from exc


def _save_revision_with_authority(
    draft: dict[str, Any],
    output: str | Path,
    *,
    authority: review_module.ReviewSourceAuthority,
    plan: dict[str, Any] | None = None,
    formal_scene: str | None = None,
) -> Path:
    try:
        destination = fs_guard.guard_output_path(output)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewUIError(f"draft output path is unsafe: {exc}") from exc
    if destination.is_relative_to(authority.corpus_root) or destination.is_relative_to(
        authority.hpe_delivery_root
    ):
        _fail("draft revisions may not be written inside corpus or HPE source roots")
    try:
        validated = review_module._validate_draft_object(
            copy.deepcopy(draft), authority=authority, plan=plan
        )
    except (review_module.TranscriptReviewError, OSError) as exc:
        raise TranscriptReviewUIError(f"draft checkpoint validation failed: {exc}") from exc
    payload = review_module._canonical_json_bytes(validated)

    def precheck() -> None:
        replayed = review_module._validate_draft_object(
            copy.deepcopy(draft), authority=authority, plan=plan
        )
        if not review_module._type_exact_json_equal(replayed, validated):
            _fail("draft or live sources changed before checkpoint publication")

    def postcheck(path: Path) -> None:
        loaded = review_module._load_review_draft_with_authority(
            path,
            authority=authority,
            plan=plan,
            formal_scene=formal_scene,
        )
        if path.read_bytes() != payload or not review_module._type_exact_json_equal(
            loaded, validated
        ):
            _fail("published draft checkpoint does not replay exactly")

    try:
        return review_module._atomic_publish_no_clobber(
            destination,
            payload,
            pre_publish_check=precheck,
            post_publish_check=postcheck,
        )
    except (review_module.TranscriptReviewError, OSError) as exc:
        raise TranscriptReviewUIError(f"draft checkpoint publication failed: {exc}") from exc


def save_review_revision(draft: dict[str, Any], output: str | Path) -> Path:
    """Save a new fully replayed production revision; never overwrite."""

    scene = review_module._require_string(draft.get("scene"), "scene")
    expected_parent = review_module.canonical_review_scene_dir(scene)
    destination = fs_guard.lexical_absolute(output)
    if (
        destination.parent != expected_parent
        or review_module._DRAFT_FILENAME_RE.fullmatch(destination.name) is None
    ):
        _fail(
            "formal draft revisions must be named draft.revNNNN.json inside "
            f"{expected_parent}"
        )
    return _save_revision_with_authority(
        draft,
        destination,
        authority=review_module.default_source_authority(),
        formal_scene=scene,
    )


def next_revision_path(current: str | Path) -> Path:
    """Return the first unused deterministic sibling revision path."""

    current_path = fs_guard.lexical_absolute(current)
    match = review_module._DRAFT_FILENAME_RE.fullmatch(current_path.name)
    if match is None:
        _fail("current production draft must be named draft.revNNNN.json")
    current_index = int(current_path.name.removeprefix("draft.rev").removesuffix(".json"))
    for index in range(current_index + 1, 10000):
        candidate = current_path.with_name(f"draft.rev{index:04d}.json")
        try:
            candidate.lstat()
        except FileNotFoundError:
            return candidate
    _fail("no free draft revision number remains")


def _event_and_clip(
    draft: dict[str, Any], source_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = [item for item in draft["events"] if item["source_index"] == source_index]
    clips = [item for item in draft["clip_plan"] if item["source_index"] == source_index]
    if len(events) != 1 or len(clips) != 1:
        _fail("draft event/clip coverage is not one-to-one")
    return events[0], clips[0]


def _sample_index(value: Any, rate: int, label: str) -> int:
    try:
        seconds = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise TranscriptReviewUIError(f"{label} is not a decimal") from exc
    if not seconds.is_finite() or seconds < 0:
        _fail(f"{label} must be finite and non-negative")
    return int((seconds * rate).to_integral_value(rounding=ROUND_CEILING))


def _read_clip_pcm16(
    item: dict[str, Any], *, label: str
) -> tuple[int, array.array[int]]:
    wav_binding = review_module._require_object(item.get("wav"), f"{label}.wav")
    review_module._require_exact_keys(
        wav_binding,
        {"path", "size_bytes", "sha256", "media_type"},
        f"{label}.wav",
    )
    if wav_binding["media_type"] != "audio/wav":
        _fail(f"{label} binding must declare audio/wav")
    path = _require_online_regular_file(Path(wav_binding["path"]), label)
    try:
        with fs_guard.guarded_read(path) as locked:
            if (
                type(wav_binding["size_bytes"]) is not int
                or locked.opening_state.size_bytes != wav_binding["size_bytes"]
            ):
                _fail(f"{label} size changed since the draft was sealed")
            digest = fs_guard.hash_guarded_read(locked)
            if digest != wav_binding["sha256"]:
                _fail(f"{label} SHA-256 changed since the draft was sealed")
            os.lseek(locked.fd, 0, os.SEEK_SET)
            with os.fdopen(locked.fd, "rb", closefd=False) as raw_handle:
                with wave.open(raw_handle, "rb") as handle:
                    if (
                        handle.getnchannels() != 1
                        or handle.getsampwidth() != 2
                        or handle.getcomptype() != "NONE"
                    ):
                        _fail(f"{label} must be mono uncompressed PCM16")
                    rate = handle.getframerate()
                    start = _sample_index(
                        item["extract_local_start_s"], rate, f"{label}.start"
                    )
                    end = _sample_index(
                        item["extract_local_end_s_exclusive"], rate, f"{label}.end"
                    )
                    if end <= start or end > handle.getnframes():
                        _fail(
                            f"{label} clip escapes the authorised physical WAV interval"
                        )
                    handle.setpos(start)
                    payload = handle.readframes(end - start)
            fs_guard.verify_guarded_read(locked)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewUIError(f"{label} guarded read failed: {exc}") from exc
    samples: array.array[int] = array.array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":  # WAV PCM is little-endian.
        samples.byteswap()
    return rate, samples


def _write_temp_pcm16(
    output_root: Path,
    *,
    prefix: str,
    rate: int,
    samples: array.array[int],
) -> Path:
    payload = array.array("h", samples)
    if sys.byteorder != "little":
        payload.byteswap()
    descriptor, raw_path = tempfile.mkstemp(
        prefix=prefix, suffix=".wav", dir=output_root
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as raw_handle:
            descriptor = -1
            with wave.open(raw_handle, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(payload.tobytes())
        return path
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def render_review_clip(
    draft: dict[str, Any],
    source_index: int,
    channel: str,
    output_dir: str | Path,
) -> Path:
    """Render one authorised temporary channel or equal-weight four-way mix."""

    _event, clip = _event_and_clip(draft, source_index)
    per_speaker = review_module._require_object(
        clip.get("per_speaker_corrected_audio_read_only"), "clip speakers"
    )
    if channel not in {*review_module.SPEAKERS, "mix"}:
        _fail("clip channel must be A/B/C/R or mix")
    try:
        output_root, _identity = fs_guard.guard_directory(output_dir)
    except fs_guard.TranscriptPathGuardError as exc:
        raise TranscriptReviewUIError(
            f"temporary clip output directory is unsafe: {exc}"
        ) from exc
    if channel != "mix":
        rate, samples = _read_clip_pcm16(
            per_speaker[channel], label=f"{channel} corrected review audio"
        )
        return _write_temp_pcm16(
            output_root,
            prefix=f"{clip['clip_id']}-{channel}-",
            rate=rate,
            samples=samples,
        )

    channels = [
        _read_clip_pcm16(per_speaker[speaker], label=f"{speaker} corrected review audio")
        for speaker in review_module.SPEAKERS
    ]
    rates = {rate for rate, _samples in channels}
    if len(rates) != 1:
        _fail("corrected review channels do not share a sample rate")
    n_samples = min(len(samples) for _rate, samples in channels)
    if n_samples <= 0:
        _fail("combined review clip is empty")
    mixed: array.array[int] = array.array("h")
    for index in range(n_samples):
        value = round(
            sum(samples[index] for _rate, samples in channels) / len(channels)
        )
        mixed.append(max(-32768, min(32767, value)))
    return _write_temp_pcm16(
        output_root,
        prefix=f"{clip['clip_id']}-{channel}-",
        rate=rates.pop(),
        samples=mixed,
    )


def apply_reviewed_event_decision(
    draft: dict[str, Any],
    source_index: int,
    *,
    speaker: str,
    start_timecode: str,
    end_timecode: str,
    note: str,
) -> None:
    """Apply one explicit human decision; display/playback never calls this."""

    event, _clip = _event_and_clip(draft, source_index)
    if speaker not in review_module.SPEAKERS:
        _fail("reviewed speaker must be A/B/C/R")
    try:
        start = acceptance_module.parse_timecode_frames(start_timecode)
        end = acceptance_module.parse_timecode_frames(end_timecode)
    except acceptance_module.TranscriptAcceptanceError as exc:
        raise TranscriptReviewUIError(str(exc)) from exc
    if end <= start:
        _fail("reviewed event must have a positive half-open interval")
    note = note.strip()
    replacements = {
        "speaker": speaker,
        "start_timecode": start_timecode,
        "end_timecode": end_timecode,
    }
    changed = any(replacements[field] != event["source"][field] for field in replacements)
    if changed and not note:
        _fail("a non-empty note is required for any manual change")
    for candidate in draft["semantic_candidates"]:
        if candidate["source_index"] != source_index:
            continue
        status = candidate["decision"]["status"]
        selected = replacements[candidate["field"]]
        if status == "unresolved":
            _fail(f"semantic candidate {candidate['candidate_id']} remains unresolved")
        if status == "accepted" and selected != candidate["replacement"]:
            _fail(f"accepted candidate {candidate['candidate_id']} must be applied")
        if status == "rejected" and selected == candidate["replacement"]:
            _fail(f"rejected candidate {candidate['candidate_id']} cannot be selected")
    event["review"] = {
        "reviewed": True,
        **{
            field: {
                "action": (
                    "keep" if replacements[field] == event["source"][field] else "change"
                ),
                "replacement": (
                    None
                    if replacements[field] == event["source"][field]
                    else replacements[field]
                ),
            }
            for field in ("speaker", "start_timecode", "end_timecode")
        },
        "note": note,
    }


def set_semantic_candidate_decision(
    draft: dict[str, Any], candidate_id: str, *, status: str, note: str
) -> None:
    """Record an explicit candidate decision without marking an event reviewed."""

    if status not in {"accepted", "rejected"}:
        _fail("candidate status must be accepted or rejected")
    note = note.strip()
    if not note:
        _fail("candidate decision requires a non-empty listening note")
    matches = [
        item for item in draft["semantic_candidates"] if item["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        _fail("candidate id is missing or not unique")
    candidate = matches[0]
    candidate["decision"] = {"status": status, "note": note}
    # A candidate decision changes the semantic context of its event.  It must
    # never leave an older event-level review looking current or make the
    # progress counter claim readiness before the reviewer applies the matching
    # field decision explicitly.
    event, _clip = _event_and_clip(draft, candidate["source_index"])
    event["review"] = review_module._blank_review()


def _reviewed_values(event: dict[str, Any]) -> tuple[str, str, str, str]:
    review = event["review"]
    values: list[str] = []
    for field in ("speaker", "start_timecode", "end_timecode"):
        decision = review[field]
        values.append(
            decision["replacement"]
            if decision["action"] == "change"
            else event["source"][field]
        )
    return values[0], values[1], values[2], review["note"]


def adjacent_review_position(
    draft: dict[str, Any], current: int, delta: int
) -> int:
    """Select a neighbouring event without mutating any review decision."""

    events = draft["events"]
    if type(current) is not int or not 0 <= current < len(events):
        _fail("current review position is out of range")
    if type(delta) is not int:
        _fail("review navigation delta must be an integer")
    return max(0, min(len(events) - 1, current + delta))


def next_priority_review_position(draft: dict[str, Any], current: int) -> int:
    """Select the next flagged/unreviewed item without marking it reviewed."""

    events = draft["events"]
    adjacent_review_position(draft, current, 0)
    order = list(range(current + 1, len(events))) + list(range(0, current + 1))
    return next(
        (
            index
            for index in order
            if events[index]["review"]["reviewed"] is not True
            and (
                events[index]["flags"]
                or any(
                    item["source_index"] == events[index]["source_index"]
                    and item["decision"]["status"] == "unresolved"
                    for item in draft["semantic_candidates"]
                )
            )
        ),
        next(
            (
                index
                for index in order
                if events[index]["review"]["reviewed"] is not True
            ),
            current,
        ),
    )


def launch_review_gui(path: str | Path) -> None:  # pragma: no cover - interactive UI
    """Launch the standard-library Tk review application."""

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise TranscriptReviewUIError("Tkinter is unavailable in this Python build") from exc
    try:
        import winsound
    except ImportError as exc:
        raise TranscriptReviewUIError("winsound playback requires Windows") from exc

    workspace = load_review_workspace(path)
    root = tk.Tk()
    root.title("VAP Transcript review — coarse reference only")
    root.geometry("980x720")
    temporary = tempfile.TemporaryDirectory(prefix="vap-transcript-review-")
    state: dict[str, Any] = {
        "path": workspace.path,
        "draft": workspace.draft,
        "position": 0,
        "saved": True,
        "loading_form": False,
        "form_dirty": False,
    }

    progress_var = tk.StringVar()
    event_var = tk.StringVar()
    flags_var = tk.StringVar()
    source_fields_var = tk.StringVar()
    text_var = tk.StringVar()
    speaker_var = tk.StringVar()
    start_var = tk.StringVar()
    end_var = tk.StringVar()
    note_var = tk.StringVar()
    candidate_id_var = tk.StringVar()
    candidate_status_var = tk.StringVar(value="unresolved")
    candidate_note_var = tk.StringVar()
    candidate_evidence_var = tk.StringVar()

    def mark_form_dirty(*_args: Any) -> None:
        if not state["loading_form"]:
            state["form_dirty"] = True

    for variable in (
        speaker_var,
        start_var,
        end_var,
        note_var,
        candidate_note_var,
    ):
        variable.trace_add("write", mark_form_dirty)

    def current_event() -> dict[str, Any]:
        return state["draft"]["events"][state["position"]]

    def candidates_for_current() -> list[dict[str, Any]]:
        source_index = current_event()["source_index"]
        return [
            item
            for item in state["draft"]["semantic_candidates"]
            if item["source_index"] == source_index
        ]

    def update_progress() -> None:
        progress = review_module.review_progress(state["draft"])
        progress_var.set(
            f"Scene {progress['scene']} | event {state['position'] + 1}/"
            f"{progress['in_scope_event_count']} | reviewed "
            f"{progress['reviewed_in_scope_events']} | candidates "
            f"{progress['resolved_semantic_candidates']}/"
            f"{progress['n_semantic_candidates']} | counted gold: NO"
        )

    def show_event() -> None:
        event = current_event()
        speaker, start, end, note = _reviewed_values(event)
        state["loading_form"] = True
        event_var.set(
            f"{event['event_id']}  source_index={event['source_index']}  "
            f"status={'REVIEWED' if event['review']['reviewed'] else 'UNREVIEWED'}"
        )
        flags_var.set("Flags: " + (", ".join(event["flags"]) or "none"))
        source_fields_var.set(
            f"Source: {event['source']['speaker']}  "
            f"[{event['source']['start_timecode']}, "
            f"{event['source']['end_timecode']})"
        )
        text_var.set(event["source"]["text"])
        speaker_var.set(speaker)
        start_var.set(start)
        end_var.set(end)
        note_var.set(note)
        candidates = candidates_for_current()
        if candidates:
            candidate = candidates[0]
            candidate_id_var.set(candidate["candidate_id"])
            candidate_status_var.set(candidate["decision"]["status"])
            candidate_note_var.set(candidate["decision"]["note"])
            candidate_evidence_var.set(
                f"Proposal: {candidate['field']} {candidate['expected']} → "
                f"{candidate['replacement']}\nEvidence: {candidate['evidence']}"
            )
        else:
            candidate_id_var.set("")
            candidate_status_var.set("unresolved")
            candidate_note_var.set("")
            candidate_evidence_var.set("No semantic candidate for this event.")
        state["loading_form"] = False
        state["form_dirty"] = False
        update_progress()

    def confirm_discard_form() -> bool:
        return not state["form_dirty"] or messagebox.askyesno(
            "Unapplied fields",
            "Discard field/candidate-note edits that were not explicitly applied?",
        )

    def play(channel: str) -> None:
        try:
            clip = render_review_clip(
                state["draft"],
                current_event()["source_index"],
                channel,
                temporary.name,
            )
            winsound.PlaySound(
                str(clip), winsound.SND_FILENAME | winsound.SND_ASYNC
            )
        except Exception as exc:  # UI converts domain errors to a visible gate.
            messagebox.showerror("Playback blocked", str(exc))

    def stop() -> None:
        winsound.PlaySound(None, 0)

    def decide_candidate(status: str) -> None:
        candidate_id = candidate_id_var.get()
        if not candidate_id:
            messagebox.showinfo("No candidate", "This event has no semantic proposal.")
            return
        try:
            set_semantic_candidate_decision(
                state["draft"],
                candidate_id,
                status=status,
                note=candidate_note_var.get(),
            )
            candidate = next(
                item
                for item in state["draft"]["semantic_candidates"]
                if item["candidate_id"] == candidate_id
            )
            if status == "accepted":
                variable = {
                    "speaker": speaker_var,
                    "start_timecode": start_var,
                    "end_timecode": end_var,
                }[candidate["field"]]
                variable.set(candidate["replacement"])
            candidate_status_var.set(status)
            state["saved"] = False
            source = current_event()["source"]
            state["form_dirty"] = status == "accepted" or any(
                (
                    speaker_var.get() != source["speaker"],
                    start_var.get() != source["start_timecode"],
                    end_var.get() != source["end_timecode"],
                    bool(note_var.get().strip()),
                )
            )
            event = current_event()
            event_var.set(
                f"{event['event_id']}  source_index={event['source_index']}  "
                "status=UNREVIEWED"
            )
            update_progress()
        except TranscriptReviewUIError as exc:
            messagebox.showerror("Candidate decision blocked", str(exc))

    def apply_decision() -> None:
        try:
            apply_reviewed_event_decision(
                state["draft"],
                current_event()["source_index"],
                speaker=speaker_var.get(),
                start_timecode=start_var.get(),
                end_timecode=end_var.get(),
                note=note_var.get(),
            )
            state["saved"] = False
            show_event()
        except TranscriptReviewUIError as exc:
            messagebox.showerror("Decision blocked", str(exc))

    def keep_source_decision() -> None:
        source = current_event()["source"]
        speaker_var.set(source["speaker"])
        start_var.set(source["start_timecode"])
        end_var.set(source["end_timecode"])
        note_var.set("")
        apply_decision()

    def move(delta: int) -> None:
        if not confirm_discard_form():
            return
        state["position"] = adjacent_review_position(
            state["draft"], state["position"], delta
        )
        show_event()

    def next_priority() -> None:
        if not confirm_discard_form():
            return
        state["position"] = next_priority_review_position(
            state["draft"], state["position"]
        )
        show_event()

    def save_revision_dialog() -> None:
        if state["form_dirty"]:
            messagebox.showerror(
                "Unapplied fields",
                "Apply the current event/candidate decision before saving a revision.",
            )
            return
        suggested = next_revision_path(state["path"])
        selected = filedialog.asksaveasfilename(
            title="Save a new draft revision (existing files are never overwritten)",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not selected:
            return
        try:
            new_path = save_review_revision(state["draft"], selected)
            state["path"] = new_path
            state["saved"] = True
            messagebox.showinfo("Saved", f"New revision:\n{new_path}")
        except TranscriptReviewUIError as exc:
            messagebox.showerror("Save blocked", str(exc))

    def open_revision_dialog() -> None:
        if not confirm_discard_form():
            return
        if not state["saved"] and not messagebox.askyesno(
            "Unsaved decisions",
            "Discard in-memory decisions since the last saved revision?",
        ):
            return
        selected = filedialog.askopenfilename(
            title="Resume a reviewed draft revision",
            initialdir=str(Path(state["path"]).parent),
            filetypes=[("JSON", "*.json")],
        )
        if not selected:
            return
        try:
            loaded = load_review_workspace(selected)
            state.update(
                {"path": loaded.path, "draft": loaded.draft, "position": 0, "saved": True}
            )
            show_event()
        except TranscriptReviewUIError as exc:
            messagebox.showerror("Resume blocked", str(exc))

    def close() -> None:
        if not confirm_discard_form():
            return
        if not state["saved"] and not messagebox.askyesno(
            "Unsaved decisions", "Close without saving a new revision?"
        ):
            return
        stop()
        temporary.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)
    ttk.Label(main, textvariable=progress_var).pack(anchor="w")
    ttk.Label(main, textvariable=event_var, font=("Segoe UI", 11, "bold")).pack(
        anchor="w", pady=(8, 0)
    )
    ttk.Label(main, textvariable=flags_var, foreground="#8a4b08").pack(anchor="w")
    ttk.Label(main, textvariable=source_fields_var).pack(anchor="w")
    ttk.Label(main, textvariable=text_var, wraplength=930).pack(
        anchor="w", pady=(8, 12)
    )

    audio = ttk.LabelFrame(main, text="Correction-mapped read-only audio")
    audio.pack(fill="x")
    for channel in (*review_module.SPEAKERS, "mix"):
        ttk.Button(
            audio,
            text=f"Play {channel.upper()}",
            command=lambda item=channel: play(item),
        ).pack(side="left", padx=4, pady=6)
    ttk.Button(audio, text="Stop", command=stop).pack(side="left", padx=4)
    ttk.Label(
        audio,
        text="Playback does not mark an event reviewed.",
        foreground="#a00000",
    ).pack(side="left", padx=12)

    form = ttk.LabelFrame(main, text="Explicit human event decision")
    form.pack(fill="x", pady=10)
    ttk.Label(form, text="Speaker").grid(row=0, column=0, sticky="w", padx=4, pady=4)
    ttk.Combobox(
        form,
        textvariable=speaker_var,
        values=review_module.SPEAKERS,
        state="readonly",
        width=8,
    ).grid(row=0, column=1, sticky="w")
    ttk.Label(form, text="Start HH:MM:SS:FF").grid(row=0, column=2, padx=4)
    ttk.Entry(form, textvariable=start_var, width=16).grid(row=0, column=3)
    ttk.Label(form, text="End HH:MM:SS:FF").grid(row=0, column=4, padx=4)
    ttk.Entry(form, textvariable=end_var, width=16).grid(row=0, column=5)
    ttk.Label(form, text="Note (required for change)").grid(
        row=1, column=0, sticky="w", padx=4
    )
    ttk.Entry(form, textvariable=note_var, width=90).grid(
        row=1, column=1, columnspan=5, sticky="ew", pady=4
    )
    ttk.Button(
        form, text="Apply reviewed decision", command=apply_decision
    ).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=6)
    ttk.Button(
        form, text="Keep all source fields", command=keep_source_decision
    ).grid(row=2, column=2, columnspan=2, sticky="w", padx=4, pady=6)

    candidate = ttk.LabelFrame(main, text="Semantic proposal (independent decision)")
    candidate.pack(fill="x")
    ttk.Label(
        candidate,
        textvariable=candidate_id_var,
        font=("Segoe UI", 9, "bold"),
    ).pack(anchor="w", padx=4, pady=(4, 0))
    ttk.Label(candidate, textvariable=candidate_status_var).pack(anchor="w", padx=4)
    ttk.Label(candidate, textvariable=candidate_evidence_var, wraplength=920).pack(
        anchor="w", padx=4, pady=4
    )
    candidate_line = ttk.Frame(candidate)
    candidate_line.pack(fill="x")
    ttk.Label(candidate_line, text="Listening note").pack(side="left", padx=4)
    ttk.Entry(candidate_line, textvariable=candidate_note_var, width=60).pack(
        side="left", padx=4
    )
    ttk.Button(
        candidate_line, text="Accept proposal", command=lambda: decide_candidate("accepted")
    ).pack(side="left", padx=4)
    ttk.Button(
        candidate_line, text="Reject proposal", command=lambda: decide_candidate("rejected")
    ).pack(side="left", padx=4)

    nav = ttk.Frame(main)
    nav.pack(fill="x", pady=12)
    ttk.Button(nav, text="Previous", command=lambda: move(-1)).pack(side="left")
    ttk.Button(nav, text="Next", command=lambda: move(1)).pack(side="left", padx=4)
    ttk.Button(nav, text="Next priority/unreviewed", command=next_priority).pack(
        side="left", padx=4
    )
    ttk.Button(nav, text="Save new revision", command=save_revision_dialog).pack(
        side="right", padx=4
    )
    ttk.Button(nav, text="Open / resume revision", command=open_revision_dialog).pack(
        side="right", padx=4
    )
    show_event()
    root.mainloop()
