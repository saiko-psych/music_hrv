"""Flet-based landing page for the Music HRV Toolkit."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import flet as ft

from music_hrv.cleaning.rr import CleaningConfig
from music_hrv.config.sections import SectionGroup
from music_hrv.io import DEFAULT_ID_PATTERN
from music_hrv.prep import PreparationSummary, load_hrv_logger_preview
from music_hrv.segments.section_normalizer import SectionNormalizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASCII_DIR = PROJECT_ROOT / "docs" / "ascii"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_HRV_LOGGER_DIR = PROJECT_ROOT / "data" / "raw" / "hrv_logger"
ANSI_COLOR_PATTERN = re.compile(r"\x1b\[38;2;(\d+);(\d+);(\d+)m")
ASCII_FONT_SIZE = 12
ASCII_MIN_FONT_SIZE = 4
ASCII_CHAR_PIXEL_RATIO = 1.05
ASCII_RESPONSIVE_SCALE = 0.8
ASCII_LINE_GAP_FACTOR = 0.45
ASCII_HORIZONTAL_PADDING = 160
ASCII_MIN_WIDTH = 180
APP_ROUTE_NAME = "music-hrv"
DEFAULT_CLEANING_CONFIG = CleaningConfig()


def load_ascii_art(name: str) -> str:
    """Read ASCII art from docs/ascii (supports .ans and .txt)."""

    for extension in (".ans", ".txt"):
        path = ASCII_DIR / f"{name}{extension}"
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Missing ASCII art asset: {name}")


def parse_ansi_art(raw: str, *, default_color: str = "#f6f5ff") -> list[list[tuple[str, str]]]:
    """Convert ANSI color escapes into line-wise spans."""

    tokens = re.split(r"(\x1b\[38;2;\d+;\d+;\d+m|\x1b\[0m|\n)", raw)
    lines: list[list[tuple[str, str]]] = []
    current_color = default_color
    buffer = []
    spans: list[tuple[str, str]] = []

    def flush_buffer() -> None:
        if buffer:
            spans.append(("".join(buffer), current_color))
            buffer.clear()

    for token in tokens:
        if not token:
            continue
        if token == "\n":
            flush_buffer()
            if spans:
                lines.append(spans.copy())
            else:
                lines.append([])
            spans.clear()
            continue
        if token == "\x1b[0m":
            flush_buffer()
            current_color = default_color
            continue
        color_match = ANSI_COLOR_PATTERN.fullmatch(token)
        if color_match:
            flush_buffer()
            r, g, b = map(int, color_match.groups())
            current_color = f"#{r:02x}{g:02x}{b:02x}"
            continue
        buffer.append(token)

    flush_buffer()
    if spans:
        lines.append(spans)
    return lines


@dataclass(slots=True)
class CustomSection:
    slug: str
    label: str
    synonyms: set[str]


class DataPrepController:
    """Stateful helper driving the data preparation tab."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        pattern: str = DEFAULT_ID_PATTERN,
        cleaning_config: CleaningConfig | None = None,
    ) -> None:
        self.current_dir = Path(root).expanduser() if root else None
        self.pattern = pattern
        self.cleaning_config = cleaning_config or DEFAULT_CLEANING_CONFIG
        self.normalizer = SectionNormalizer.from_yaml()
        default_required = tuple(
            definition.name
            for definition in self.normalizer.config.iter_definitions()
            if definition.required
        )
        self.section_order = tuple(
            definition.name for definition in self.normalizer.config.iter_definitions()
        )
        self.status_message = "Select a folder to run data preparation."
        self.group_templates = self._init_group_templates(default_required)
        self.default_group_key = next(iter(self.group_templates))
        self.default_required_sections = (
            default_required
            if default_required
            else self.group_templates[self.default_group_key].required_sections
        )
        self.group_assignments: dict[str, str] = {}
        self.summaries: dict[str, PreparationSummary] = {}
        self.selected_participant: str | None = None
        self.custom_sections: dict[str, CustomSection] = {}

    def _init_group_templates(
        self, default_required: tuple[str, ...]
    ) -> dict[str, SectionGroup]:
        if self.normalizer.config.groups:
            return dict(self.normalizer.config.groups)
        fallback = SectionGroup(
            name="default",
            label="Default protocol",
            required_sections=default_required or (),
        )
        return {"default": fallback}

    def set_directory(self, new_dir: Path | str | None) -> None:
        self.current_dir = Path(new_dir).expanduser() if new_dir else None

    def scan(self) -> None:
        if not self.current_dir:
            self.summaries = {}
            self.selected_participant = None
            self.status_message = "Select a folder to run data preparation."
            return
        summaries = load_hrv_logger_preview(
            self.current_dir,
            pattern=self.pattern,
            config=self.cleaning_config,
            normalizer=self.normalizer,
        )
        self.summaries = {summary.participant_id: summary for summary in summaries}
        self.group_assignments = {
            participant: self.group_assignments.get(participant, self.default_group_key)
            for participant in self.summaries
        }
        if self.summaries:
            if (
                not self.selected_participant
                or self.selected_participant not in self.summaries
            ):
                self.selected_participant = sorted(self.summaries)[0]
            self.status_message = (
                f"{len(self.summaries)} participant(s) loaded from {self._relative_dir_label()}"
            )
        else:
            self.selected_participant = None
            if self.current_dir.exists():
                self.status_message = (
                    f"No HRV Logger files found in {self._relative_dir_label()}"
                )
            else:
                self.status_message = f"Folder not found: {self.current_dir}"

    def _relative_dir_label(self) -> str:
        if not self.current_dir:
            return "Not selected"
        try:
            return str(self.current_dir.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.current_dir)

    def current_dir_label(self) -> str:
        return self._relative_dir_label()

    def participants(self) -> list[str]:
        return sorted(self.summaries)

    def sorted_summaries(self) -> list[PreparationSummary]:
        return [self.summaries[pid] for pid in self.participants()]

    def available_groups(self) -> list[tuple[str, SectionGroup]]:
        return list(self.group_templates.items())

    def group_lookup(self) -> dict[str, SectionGroup]:
        return dict(self.group_templates)

    def expected_sections(self, participant_id: str) -> tuple[str, ...]:
        key = self.group_assignments.get(participant_id, self.default_group_key)
        template = self.group_templates.get(key)
        if template and template.required_sections:
            return template.required_sections
        return self.default_required_sections

    def missing_sections(self, participant_id: str) -> tuple[str, ...]:
        summary = self.summaries.get(participant_id)
        if not summary:
            return ()
        expected = self.expected_sections(participant_id)
        return tuple(section for section in expected if section not in summary.present_sections)

    def select_participant(self, participant_id: str | None) -> None:
        if participant_id and participant_id in self.summaries:
            self.selected_participant = participant_id
        else:
            self.selected_participant = None

    def set_group(self, participant_id: str, group_key: str) -> None:
        if participant_id in self.summaries and group_key in self.group_templates:
            self.group_assignments[participant_id] = group_key

    def rename_event(self, participant_id: str, index: int, new_label: str) -> None:
        summary = self.summaries.get(participant_id)
        if not summary or index >= len(summary.events):
            return
        summary.events[index].raw_label = new_label
        canonical = self.normalizer.normalize(new_label)
        if not canonical:
            canonical = self.match_custom_section(new_label)
        summary.events[index].canonical = canonical
        summary.present_sections = {
            status.canonical for status in summary.events if status.canonical
        }

    def section_label(self, canonical: str | None) -> str:
        if not canonical:
            return "unknown"
        if canonical in self.custom_sections:
            return self.custom_sections[canonical].label
        definition = self.normalizer.definition_for(canonical)
        if definition and definition.description:
            return definition.description
        return canonical.replace("_", " ").title()

    def has_data(self) -> bool:
        return bool(self.summaries)

    def available_section_labels(self) -> list[tuple[str, str]]:
        labels = [(name, self.section_label(name)) for name in self.section_order]
        labels.extend((slug, section.label) for slug, section in self.custom_sections.items())
        return labels

    def update_group_template(self, group_key: str, sections: list[str]) -> None:
        template = self.group_templates.get(group_key)
        if not template:
            return
        self.group_templates[group_key] = SectionGroup(
            name=template.name,
            label=template.label,
            required_sections=tuple(sections),
        )

    def create_group(
        self,
        name: str,
        label: str | None = None,
        sections: list[str] | None = None,
    ) -> str:
        key = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
        if not key:
            key = f"group-{len(self.group_templates) + 1}"
        if key in self.group_templates:
            suffix = 2
            while f"{key}-{suffix}" in self.group_templates:
                suffix += 1
            key = f"{key}-{suffix}"
        clean_label = label.strip() if label else name.strip().title() or key.title()
        self.group_templates[key] = SectionGroup(
            name=key,
            label=clean_label,
            required_sections=tuple(sections or self.default_required_sections),
        )
        return key

    def _slugify_custom(self, label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower())
        slug = slug.strip("-")
        return slug or f"section-{len(self.custom_sections) + 1}"

    def add_custom_section(self, label: str, synonyms: list[str] | None = None) -> str:
        slug = self._slugify_custom(label)
        suffix = 2
        while slug in self.custom_sections:
            slug = f"{slug}-{suffix}"
            suffix += 1
        synonyms_set = {self._normalize_synonym(value) for value in (synonyms or []) if value.strip()}
        self.custom_sections[slug] = CustomSection(
            slug=slug,
            label=label.strip() or slug.title(),
            synonyms=synonyms_set,
        )
        self.reconcile_custom_sections()
        return slug

    def _normalize_synonym(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    def match_custom_section(self, raw_label: str) -> str | None:
        normalized = self._normalize_synonym(raw_label)
        slug = self._slugify_custom(raw_label)
        for section in self.custom_sections.values():
            if normalized in section.synonyms or slug == section.slug:
                section.synonyms.add(normalized)
                return section.slug
        return None

    def reconcile_custom_sections(self) -> None:
        for summary in self.summaries.values():
            for event in summary.events:
                if not event.canonical:
                    event.canonical = self.match_custom_section(event.raw_label)
            summary.present_sections = {
                status.canonical for status in summary.events if status.canonical
            }

    def set_event_canonical(
        self, participant_id: str, index: int, canonical: str | None
    ) -> None:
        summary = self.summaries.get(participant_id)
        if not summary or index >= len(summary.events):
            return
        summary.events[index].canonical = canonical or None
        if canonical in self.custom_sections:
            normalized = self._normalize_synonym(summary.events[index].raw_label)
            self.custom_sections[canonical].synonyms.add(normalized)
        summary.present_sections = {
            status.canonical for status in summary.events if status.canonical
        }

    def canonical_options(self) -> list[tuple[str | None, str]]:
        options = [(None, "— Unassigned —")]
        options.extend((value, label) for value, label in self.available_section_labels())
        return options

    def _slugify_custom(self, label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower())
        slug = slug.strip("-")
        return slug or f"section-{len(self.custom_sections) + 1}"

    def add_custom_section(self, label: str) -> str:
        slug = self._slugify_custom(label)
        suffix = 2
        while slug in self.custom_sections:
            slug = f"{slug}-{suffix}"
            suffix += 1
        self.custom_sections[slug] = label.strip() or slug.title()
        return slug



def build_data_prep_panel(page: ft.Page) -> ft.Column:
    """Create the interactive panel that previews data preparation output."""

    controller = DataPrepController()

    status_text = ft.Text(
        controller.status_message,
        color=ft.colors.GREY_400,
        size=12,
    )
    folder_input = ft.TextField(
        value=str(controller.current_dir or DATA_HRV_LOGGER_DIR),
        label="Folder path",
        width=400,
        filled=True,
    )
    table_heading = ft.Text(
        "Participant metrics",
        size=16,
        weight=ft.FontWeight.W_600,
    )
    table_description = ft.Text(
        "Each row summarises RR cleaning stats. Assign a protocol group to check expected events.",
        size=12,
        color=ft.colors.GREY_500,
    )
    table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Participant")),
            ft.DataColumn(ft.Text("Beats (raw)")),
            ft.DataColumn(ft.Text("Beats (clean)")),
            ft.DataColumn(ft.Text("Artifacts")),
            ft.DataColumn(ft.Text("Duration")),
            ft.DataColumn(ft.Text("Events")),
            ft.DataColumn(ft.Text("Missing")),
            ft.DataColumn(ft.Text("Group")),
        ],
        rows=[],
        column_spacing=18,
        data_row_min_height=60,
    )
    event_panel = ft.Container(
        content=ft.Text(
            "Select a participant to inspect events.",
            color=ft.colors.GREY_500,
        ),
        padding=10,
        bgcolor=ft.colors.with_opacity(0.03, ft.colors.WHITE),
        border_radius=8,
    )
    group_editor_panel = ft.Container(
        padding=10,
        bgcolor=ft.colors.with_opacity(0.03, ft.colors.WHITE),
        border_radius=8,
    )
    selected_group_key = controller.default_group_key
    NONE_OPTION_KEY = "__none__"

    def canonical_dropdown_options() -> list[ft.dropdown.Option]:
        return [
            ft.dropdown.Option(
                key=NONE_OPTION_KEY if value is None else value,
                text=label,
            )
            for value, label in controller.canonical_options()
        ]

    def rebuild_table() -> None:
        rows: list[ft.DataRow] = []
        group_options = [
            ft.dropdown.Option(key=group_key, text=template.label)
            for group_key, template in controller.available_groups()
        ]
        for summary in controller.sorted_summaries():
            pid = summary.participant_id
            missing = controller.missing_sections(pid)
            group_value = controller.group_assignments.get(pid, controller.default_group_key)

            def _make_group_dropdown(participant: str, value: str) -> ft.Dropdown:
                return ft.Container(
                    width=380,
                    content=ft.Dropdown(
                        value=value,
                        width=360,
                        options=group_options,
                        on_change=lambda e, participant=participant: handle_group_change(
                            participant, e.data
                        ),
                    ),
                )

            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(pid)),
                        ft.DataCell(ft.Text(str(summary.total_beats))),
                        ft.DataCell(ft.Text(str(summary.retained_beats))),
                        ft.DataCell(ft.Text(f"{summary.artifact_ratio * 100:.1f}%")),
                        ft.DataCell(ft.Text(f"{summary.duration_s / 60:.1f} min")),
                        ft.DataCell(ft.Text(str(summary.events_detected))),
                        ft.DataCell(
                            ft.Text(
                                "✓" if not missing else f"{len(missing)} missing",
                                color=ft.colors.GREEN_400
                                if not missing
                                else ft.colors.AMBER_400,
                            )
                        ),
                        ft.DataCell(_make_group_dropdown(pid, group_value)),
                    ],
                    selected=controller.selected_participant == pid,
                    on_select_changed=lambda e, participant=pid: handle_row_select(
                        participant
                    ),
                )
            )
        table.rows = rows
        if table.page:
            table.update()

    def rebuild_group_editor() -> None:
        nonlocal selected_group_key
        if selected_group_key not in controller.group_lookup():
            selected_group_key = controller.default_group_key
        current_template = controller.group_templates[selected_group_key]
        checkboxes: list[ft.Control] = []

        def handle_toggle(section_name: str, value: bool) -> None:
            template_sections = set(controller.group_templates[selected_group_key].required_sections)
            if value:
                template_sections.add(section_name)
            else:
                template_sections.discard(section_name)
            controller.update_group_template(selected_group_key, sorted(template_sections))
            rebuild_table()
            refresh_event_panel()
            rebuild_group_editor()

        for section_name, label in controller.available_section_labels():
            checkbox = ft.Checkbox(
                label=label,
                value=section_name in current_template.required_sections,
                on_change=lambda e, section=section_name: handle_toggle(section, bool(e.control.value)),
            )
            checkboxes.append(checkbox)

        create_name = ft.TextField(label="Protocol ID", width=160, hint_text="e.g. music_a")
        create_label = ft.TextField(label="Display name", width=200, hint_text="Music A")
        new_section_field = ft.TextField(label="New section", width=220, hint_text="e.g. music block a")
        new_section_synonyms = ft.TextField(
            label="Accepted raw labels",
            width=280,
            hint_text="Comma-separated synonyms",
        )

        def handle_create_group(_: ft.ControlEvent) -> None:
            nonlocal selected_group_key
            if not create_name.value:
                create_name.error_text = "Provide an identifier"
                if create_name.page:
                    create_name.update()
                return
            new_key = controller.create_group(
                create_name.value,
                label=create_label.value or create_name.value,
                sections=list(controller.group_templates[selected_group_key].required_sections),
            )
            create_name.value = ""
            create_label.value = ""
            selected_group_key = new_key
            rebuild_table()
            refresh_event_panel()
            rebuild_group_editor()
            page.update()

        def handle_add_section(_: ft.ControlEvent) -> None:
            if not new_section_field.value:
                new_section_field.error_text = "Provide a label"
                if new_section_field.page:
                    new_section_field.update()
                return
            synonyms = []
            if new_section_synonyms.value:
                synonyms = [value.strip() for value in new_section_synonyms.value.split(",") if value.strip()]
            slug = controller.add_custom_section(new_section_field.value, synonyms=synonyms)
            template_sections = list(controller.group_templates[selected_group_key].required_sections)
            if slug not in template_sections:
                template_sections.append(slug)
            controller.update_group_template(selected_group_key, template_sections)
            new_section_field.value = ""
            new_section_synonyms.value = ""
            rebuild_table()
            refresh_event_panel()
            rebuild_group_editor()

        group_editor_panel.content = ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(
                            f"Group “{current_template.label}” required sections",
                            size=14,
                            weight=ft.FontWeight.W_600,
                        ),
                        ft.Dropdown(
                            value=selected_group_key,
                            width=260,
                            options=[
                                ft.dropdown.Option(key=key, text=template.label)
                                for key, template in controller.available_groups()
                            ],
                            on_change=lambda e: select_group_for_edit(e.data),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Text(
                    "Toggle the canonical sections that must appear for participants assigned to this group.",
                    size=12,
                    color=ft.colors.GREY_500,
                ),
                ft.Column(checkboxes, spacing=4),
                ft.Divider(height=20),
                ft.Text(
                    "Create a new protocol",
                    size=13,
                    weight=ft.FontWeight.W_600,
                ),
                ft.Row(
                    [
                        create_name,
                        create_label,
                        ft.ElevatedButton(
                            "Add protocol",
                            icon=ft.icons.ADD,
                            on_click=handle_create_group,
                        ),
                    ],
                    spacing=10,
                ),
                ft.Divider(height=20),
                ft.Text(
                    "Add custom section",
                    size=13,
                    weight=ft.FontWeight.W_600,
                ),
                ft.Row(
                    [
                        new_section_field,
                        new_section_synonyms,
                        ft.ElevatedButton(
                            "Add section",
                            icon=ft.icons.LABEL,
                            on_click=handle_add_section,
                        ),
                    ],
                    spacing=10,
                ),
            ],
            spacing=8,
        )
        if group_editor_panel.page:
            group_editor_panel.update()

    def select_group_for_edit(new_key: str | None) -> None:
        nonlocal selected_group_key
        if new_key and new_key in controller.group_lookup():
            selected_group_key = new_key
            rebuild_group_editor()

    def refresh_event_panel() -> None:
        pid = controller.selected_participant
        if not pid or pid not in controller.summaries:
            message = (
                "Choose a folder and scan data to inspect events."
                if not controller.has_data()
                else "Select a participant to inspect events."
            )
            event_panel.content = ft.Text(
                message,
                color=ft.colors.GREY_500,
            )
            if event_panel.page:
                event_panel.update()
            return
        summary = controller.summaries[pid]
        expected = controller.expected_sections(pid)
        missing = set(controller.missing_sections(pid))
        chips = []
        for canonical in expected:
            chips.append(
                ft.Chip(
                    label=ft.Text(controller.section_label(canonical)),
                    bgcolor=ft.colors.with_opacity(
                        0.15, ft.colors.GREEN_300 if canonical not in missing else ft.colors.RED_200
                    ),
                    leading=ft.Icon(
                        ft.icons.CHECK
                        if canonical not in missing
                        else ft.icons.ERROR_OUTLINE,
                        size=14,
                        color=ft.colors.WHITE,
                    ),
                )
            )
        if not chips:
            chips_text = ft.Text(
                "No required events defined for the selected group.",
                size=12,
                color=ft.colors.GREY_500,
            )
        else:
            chips_text = ft.ResponsiveRow(
                columns=12,
                controls=[
                    ft.Container(control, col={"xs": 12, "sm": 6, "md": 4, "lg": 3})
                    for control in chips
                ],
            )

        def handle_event_edit(participant: str, index: int, value: str) -> None:
            controller.rename_event(participant, index, value)
            rebuild_table()
            refresh_event_panel()

        event_rows = []
        dropdown_options = canonical_dropdown_options()
        for idx, event in enumerate(summary.events):
            desired_width = min(720, max(320, len(event.raw_label) * 9))
            event_rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(idx + 1))),
                        ft.DataCell(
                            ft.Container(
                                width=desired_width + 20,
                                content=ft.TextField(
                                    value=event.raw_label,
                                    width=desired_width,
                                    height=52,
                                    multiline=len(event.raw_label) > 50,
                                    border_radius=6,
                                    text_size=13,
                                    content_padding=ft.padding.symmetric(vertical=8, horizontal=12),
                                    border_color=ft.colors.with_opacity(0.2, ft.colors.WHITE),
                                    on_change=lambda e, p=pid, i=idx: handle_event_edit(
                                        p, i, e.control.value  # type: ignore[arg-type]
                                    ),
                                ),
                            )
                        ),
                        ft.DataCell(
                            ft.Dropdown(
                                value=event.canonical or NONE_OPTION_KEY,
                                width=360,
                                options=dropdown_options,
                                on_change=lambda e, p=pid, i=idx: handle_event_canonical(
                                    p, i, e.data
                                ),
                            )
                        ),
                    ]
                )
            )

        events_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("#")),
                ft.DataColumn(ft.Text("Raw label")),
                ft.DataColumn(ft.Text("Canonical")),
            ],
            rows=event_rows,
            column_spacing=16,
            data_row_min_height=40,
        )

        event_panel.content = ft.Column(
            [
                ft.Text(
                    f"Events for {pid}",
                    size=16,
                    weight=ft.FontWeight.W_600,
                ),
                chips_text,
                ft.Text(
                    "Edit event labels to align them with the expected sections. Changes are stored locally.",
                    size=12,
                    color=ft.colors.GREY_500,
                ),
                events_table,
            ],
            spacing=10,
        )
        if event_panel.page:
            event_panel.update()

    def handle_row_select(participant: str) -> None:
        controller.select_participant(participant)
        rebuild_table()
        refresh_event_panel()

    def handle_group_change(participant: str, value: str | None) -> None:
        if value:
            controller.set_group(participant, value)
            rebuild_table()
            refresh_event_panel()

    def handle_event_canonical(participant: str, index: int, key: str | None) -> None:
        canonical = None if not key or key == NONE_OPTION_KEY else key
        controller.set_event_canonical(participant, index, canonical)
        rebuild_table()
        refresh_event_panel()

    def handle_directory_pick(result: ft.FilePickerResultEvent) -> None:
        path = result.path or (result.paths[0] if result.paths else None)
        if path:
            folder_input.value = path
            apply_manual_folder()

    def apply_manual_folder(_: ft.ControlEvent | None = None) -> None:
        raw_value = folder_input.value.strip()
        if not raw_value:
            status_text.value = "Please provide a folder path."
            page.update()
            return
        controller.set_directory(Path(raw_value))
        controller.scan()
        folder_input.value = str(controller.current_dir or raw_value)
        status_text.value = controller.status_message
        rebuild_table()
        refresh_event_panel()
        rebuild_group_editor()
        page.update()

    def handle_refresh(_: ft.ControlEvent | None = None) -> None:
        apply_manual_folder()

    directory_picker = ft.FilePicker(on_result=handle_directory_pick)
    page.overlay.append(directory_picker)
    page.update()
    rebuild_table()
    refresh_event_panel()
    rebuild_group_editor()

    def launch_directory_picker(_: ft.ControlEvent | None = None) -> None:
        if page.web:
            status_text.value = "Folder picker not available in browser — paste a path and click Scan."
            page.update()
            return
        initial = controller.current_dir or DATA_HRV_LOGGER_DIR
        try:
            directory_picker.get_directory_path(initial_directory=str(initial))
        except Exception:
            status_text.value = "Folder picker failed — paste a path and click Scan."
            page.update()

    choose_button = ft.FilledTonalButton(
        "Choose folder",
        icon=ft.icons.FOLDER_OPEN,
        on_click=launch_directory_picker,
    )
    manual_button = ft.ElevatedButton(
        "Scan folder",
        icon=ft.icons.SEARCH,
        on_click=apply_manual_folder,
    )
    refresh_button = ft.IconButton(
        icon=ft.icons.REFRESH,
        tooltip="Rescan current folder",
        on_click=handle_refresh,
    )

    return ft.Column(
        [
            ft.Row(
                [
                    ft.Text(
                        "Data preparation preview",
                        size=20,
                        weight=ft.FontWeight.W_600,
                    ),
                    refresh_button,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            ft.Row(
                [
                    folder_input,
                    choose_button,
                    manual_button,
                ],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            status_text,
            table_heading,
            table_description,
            ft.Container(
                content=table,
                bgcolor=ft.colors.with_opacity(0.05, ft.colors.WHITE),
                border_radius=8,
                padding=10,
            ),
            ft.Text(
                "Group templates",
                size=16,
                weight=ft.FontWeight.W_600,
            ),
            group_editor_panel,
            event_panel,
        ],
        spacing=12,
        expand=True,
    )

def main(page: ft.Page) -> None:
    """Render a Hyperpop / Hardtekk-flavoured landing page."""

    page.title = "Music HRV Toolkit"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#04010d"
    page.scroll = "adaptive"
    page.padding = 10
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    raw_art = load_ascii_art("mobius_main")
    ascii_lines = parse_ansi_art(raw_art)
    max_line_chars = max(
        (sum(len(span_text) for span_text, _ in line) for line in ascii_lines), default=0
    )
    baseline_width = max_line_chars * ASCII_CHAR_PIXEL_RATIO * ASCII_FONT_SIZE

    def derive_font_size(target_width: float) -> float:
        if not baseline_width or target_width <= 0:
            return ASCII_FONT_SIZE
        scale = (target_width / baseline_width) * ASCII_RESPONSIVE_SCALE
        dynamic_size = ASCII_FONT_SIZE * scale
        return max(ASCII_MIN_FONT_SIZE, min(ASCII_FONT_SIZE, dynamic_size))

    def compute_ascii_layout(target_width: float) -> tuple[list[ft.Control], float]:
        usable_width = max(target_width - ASCII_HORIZONTAL_PADDING, ASCII_MIN_WIDTH)
        font_size = derive_font_size(usable_width)
        line_gap = max(int(font_size * ASCII_LINE_GAP_FACTOR), 4)
        controls: list[ft.Control] = []

        for line in ascii_lines:
            if not line:
                controls.append(ft.Container(height=line_gap))
                continue
            spans = [
                ft.TextSpan(
                    text=span_text,
                    style=ft.TextStyle(
                        color=color,
                        font_family="RobotoMono",
                        size=font_size,
                        weight=ft.FontWeight.W_600,
                    ),
                )
                for span_text, color in line
            ]
            controls.append(
                ft.Text(
                    spans=spans,
                    text_align=ft.TextAlign.CENTER,
                    width=usable_width,
                    no_wrap=True,
                    selectable=True,
                    style=ft.TextStyle(size=font_size),
                )
            )
        return controls, usable_width

    def build_ascii_column(target_width: float) -> ft.Column:
        controls, usable_width = compute_ascii_layout(target_width)
        return ft.Column(
            controls,
            spacing=0,
            width=usable_width,
            adaptive=True,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def measured_width(candidate: float | None) -> float:
        width = candidate or page.window_width or page.width or 1200.0
        try:
            width = float(width)
        except (TypeError, ValueError):
            width = 1200.0
        return max(ASCII_MIN_WIDTH, min(width, 1600.0))

    initial_width = measured_width(page.window_width)
    hero_ascii_container = ft.Container(
        alignment=ft.alignment.center,
        width=initial_width,
        content=build_ascii_column(initial_width),
        padding=ft.padding.only(top=10, bottom=20),
    )

    def parse_event_width(raw: str | None) -> float | None:
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
            width_value = payload.get("width")
            try:
                return float(width_value)
            except (TypeError, ValueError):
                return None

    def refresh_ascii(width: float | None) -> None:
        target = measured_width(width)
        hero_ascii_container.width = target
        hero_ascii_container.content = build_ascii_column(target)
        page.update()

    def handle_resize(event: ft.ControlEvent) -> None:
        event_width = parse_event_width(getattr(event, "data", None))
        refresh_ascii(event_width or page.window_width)

    page.on_resize = handle_resize

    social_row = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10,
        controls=[
            ft.IconButton(
                icon=ft.icons.CODE,
                tooltip="GitHub: saiko-psych",
                url="https://github.com/saiko-psych",
                icon_color="#ff4dff",
            ),
            ft.IconButton(
                icon=ft.icons.EMAIL,
                tooltip="david.matischek@edu.uni-graz.at",
                url="mailto:david.matischek@edu.uni-graz.at",
                icon_color="#05f0ff",
            ),
        ],
    )

    page.snack_bar = ft.SnackBar(ft.Text("CLI integration coming soon — stay tuned!"))

    landing_view = ft.Column(
        [
            hero_ascii_container,
            social_row,
        ],
        spacing=20,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    data_prep_panel = build_data_prep_panel(page)

    tabs = ft.Tabs(
        animation_duration=300,
        expand=1,
        tabs=[
            ft.Tab(text="Home", icon=ft.icons.STAR, content=landing_view),
            ft.Tab(text="data prep", icon=ft.icons.TABLE_CHART, content=data_prep_panel),
        ],
    )

    page.add(tabs)
    refresh_ascii(page.window_width)


def run(view: ft.AppView | None = None) -> None:
    """Launch the Flet app (browser view to avoid native deps by default)."""

    ft.app(
        target=main,
        view=view or ft.AppView.WEB_BROWSER,
        assets_dir=str(ASSETS_DIR),
        name=APP_ROUTE_NAME,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
