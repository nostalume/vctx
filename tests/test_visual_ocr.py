from __future__ import annotations

from vctx.transforms.visual_ocr import _rapidocr_text


class _RapidOcrOutput:
    def __init__(self, txts: tuple[str, ...]) -> None:
        self.txts = txts


class _LegacyTextBlock:
    text = "legacy block text should not be parsed"


def test_rapidocr_text_reads_documented_output_txts() -> None:
    result = _RapidOcrOutput(("正品促销", "40°C深度防冻不结冰"))

    assert _rapidocr_text(result) == "正品促销\n40°C深度防冻不结冰"


def test_rapidocr_text_preserves_native_dict_blocks() -> None:
    result = [
        {"text": "精度 速度 メモリ使用量"},
        {"text": "模型读取字幕"},
    ]

    assert _rapidocr_text(result) == "精度 速度 メモリ使用量\n模型读取字幕"


def test_rapidocr_text_reads_tuple_result_blocks() -> None:
    result = (
        [
            [[0, 0], [1, 1], "画面の表"],
            ((0, 0), (1, 1), "処理フロー"),
        ],
        {"elapsed": 0.1},
    )

    assert _rapidocr_text(result) == "画面の表\n処理フロー"


def test_rapidocr_text_ignores_unknown_or_empty_shapes() -> None:
    assert _rapidocr_text(None) == ""
    assert _rapidocr_text({"text": "not a block list"}) == ""
    assert _rapidocr_text(([{"text": 123}, [0.1, 0.2]],)) == ""
    assert _rapidocr_text([_LegacyTextBlock()]) == ""
