"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.layers.citation_checker import on_one_line
from harness.middleware import Middleware


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            claims = []
        kept, dropped = [], 0
        for claim in claims:
            text = claim.get("text") if isinstance(claim, dict) else None
            if not isinstance(text, str) or not text:
                dropped += 1
            elif ctx.saw(text):
                kept.append(claim)  # có căn cứ: giữ nguyên, KHÔNG sửa chữ
            elif trimmed := self._trim_to_evidence(ctx, claim, text):
                kept.append(trimmed)  # chỉ CẮT bớt hai đầu, không đổi chữ
            elif halves := self._split_glued(ctx, text):
                kept.extend(halves)
                report["abstain"] = True  # hai nguồn mâu thuẫn
            else:
                dropped += 1  # bịa: bỏ đi
        ctx.state["critic_dropped"] = dropped
        report["claims"] = kept
        if not kept:
            report["abstain"] = True
            report["answer"] = (
                "Không đủ căn cứ trong các tài liệu đã đọc để trả lời câu hỏi này."
            )
        report["citations"] = sorted({c["doc_id"] for c in kept if c.get("doc_id")})
        return report

    @staticmethod
    def _trim_to_evidence(ctx, claim, text):
        """Model thật hay bọc claim trong nháy, in đậm hoặc thêm dấu chấm cuối.

        Cắt các ký tự đó ở HAI ĐẦU — phần còn lại là substring của chữ mô
        hình viết, nên vẫn giữ provenance (README §7.1: cắt thì được).
        """
        core = text.strip(TRIM_CHARS)
        if core == text or len(core) < MIN_TRIM_CHARS or not ctx.saw(core):
            return None
        doc = ctx.corpus.get(claim.get("doc_id")) if ctx.corpus is not None else None
        doc_id = claim.get("doc_id")
        if doc is None or not on_one_line(core, doc.body):
            # citation_checker chạy trước, khi chữ chưa cắt nên chưa khớp dòng nào.
            doc_id = _source(ctx, core) or doc_id
        return {**claim, "text": core, "doc_id": doc_id}

    @staticmethod
    def _split_glued(ctx, text):
        """Tách câu ghép "A và B" thành hai claim, mỗi nửa thuộc một tài liệu khác nhau.

        Cắt ĐÚNG tại vị trí liên từ chứ không `split()`: nửa sau có thể tự
        bắt đầu bằng "và …" ("… mỗi tuần và và chỉ được …").
        """
        pos = text.find(GLUE)
        while pos != -1:
            left, right = text[:pos], text[pos + len(GLUE):]
            if ctx.saw(left) and ctx.saw(right):
                left_doc, right_doc = _source(ctx, left), _source(ctx, right)
                if left_doc and right_doc and left_doc != right_doc:
                    return [
                        {"text": left, "doc_id": left_doc},
                        {"text": right, "doc_id": right_doc},
                    ]
            pos = text.find(GLUE, pos + 1)
        return None


#: Liên từ mô hình dùng để dán hai nửa câu của hai tài liệu khác nhau.
GLUE = " và "

#: Ký tự bao quanh mà model thật hay thêm vào claim: khoảng trắng, nháy,
#: in đậm/markdown, dấu câu cuối.
TRIM_CHARS = " \t\r\n\"'“”‘’«»*_`.,;:!?…"

#: Ngắn hơn mức này thì phần còn lại không còn là một khẳng định có nghĩa.
MIN_TRIM_CHARS = 20


def _source(ctx, text):
    """doc_id của tài liệu đã quan sát trọn vẹn có một dòng chứa `text`."""
    if ctx.corpus is None:
        return None
    observed = ctx.observed_text
    for doc in ctx.corpus.docs:
        if doc.body in observed and on_one_line(text, doc.body):
            return doc.doc_id
    return None
