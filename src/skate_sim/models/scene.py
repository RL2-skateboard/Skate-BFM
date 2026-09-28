from __future__ import annotations

from .board import BoardStyle, build_board_xml


def board_scene(style: BoardStyle) -> str:
    return build_board_xml(style, gallery=False)


def gallery_scene(style: BoardStyle) -> str:
    return build_board_xml(style, gallery=True)
