"""Sổ dòng dõi (Lineage) chỉ-ghi-thêm (append-only), có chuỗi băm SHA-256.
Mỗi mục chứa băm của mục trước; sửa/xóa mục cũ sẽ làm `verify` báo lỗi."""
import hashlib, sys, datetime, re, os
PATH = os.path.join(os.path.dirname(__file__), "..", "runs", "Lineage.md")
GENESIS = "0" * 64
HEAD = "# Lineage — sổ dòng dõi thực nghiệm TNR (chỉ ghi thêm)\n\nQuy ước: mỗi mục chỉ được THÊM vào cuối tệp. Băm của mục = SHA-256(băm_trước + nội_dung_mục). Kiểm tra: `python src/lineage.py verify`.\nTrạng thái hợp lệ: ĐẠT (PASS) · KHÔNG ĐẠT (FAIL) · CHƯA KẾT LUẬN (UNRESOLVED) · THĂM DÒ (không mang tính bằng chứng).\n\n"


def _entries(txt):
    return re.findall(r"(## \[L\d+\].*?)\n<!-- prev:(\w+) hash:(\w+) -->\n", txt, flags=re.S)


def _last_hash(txt):
    e = _entries(txt)
    return e[-1][2] if e else GENESIS


def append(title, status, body):
    txt = open(PATH, encoding="utf-8").read() if os.path.exists(PATH) else HEAD
    n = len(_entries(txt)) + 1
    prev = _last_hash(txt)
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    content = f"## [L{n:03d}] {title}\n- Thời điểm: {ts}\n- Trạng thái: **{status}**\n\n{body.strip()}\n"
    h = hashlib.sha256((prev + content).encode()).hexdigest()
    with open(PATH, "a", encoding="utf-8") as f:
        if not os.path.exists(PATH) or os.path.getsize(PATH) == 0:
            f.write(HEAD)
        f.write(content + f"<!-- prev:{prev} hash:{h} -->\n\n")
    return n, h


def verify():
    txt = open(PATH, encoding="utf-8").read()
    prev = GENESIS
    for content, p, h in _entries(txt):
        c = content + "\n"
        if p != prev or hashlib.sha256((prev + c).encode()).hexdigest() != h:
            print("LỖI chuỗi băm tại", content.split("\n")[0]); return False
        prev = h
    print("Chuỗi băm hợp lệ,", len(_entries(txt)), "mục"); return True


if __name__ == "__main__":
    if sys.argv[1] == "verify": sys.exit(0 if verify() else 1)
