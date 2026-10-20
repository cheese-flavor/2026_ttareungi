"""Restore the three public Seoul daily-rental CSVs; keep datasets untracked."""
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://data.seoul.go.kr/dataList/OA-14994/F/1/datasetView.do"
ENDPOINT = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
FILES = [("25", "25.1-6", 181), ("26", "25.7-12", 184), ("27", "26.1-6", 181)]


def main():
    folder = ROOT / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for seq, period, expected_rows in FILES:
        name = f"서울특별시 공공자전거 일별 대여건수_{period}.csv"
        path = folder / name
        if path.exists():
            content = path.read_bytes()
            downloaded = False
        else:
            request = Request(ENDPOINT, data=urlencode({
                "infId": "OA-14994", "seqNo": "", "seq": seq, "infSeq": "1"
            }).encode(), headers={"Referer": SOURCE, "User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                content = response.read()
            downloaded = True
        rows = None
        for encoding in ("utf-8-sig", "cp949"):
            try:
                candidate = list(csv.DictReader(io.StringIO(content.decode(encoding))))
            except UnicodeDecodeError:
                continue
            if candidate and {"대여일자", "대여건수"} <= set(candidate[0]):
                rows = candidate
                break
        if rows is None or len(rows) != expected_rows:
            raise ValueError(f"Unexpected CSV schema or row count: {name}; file not overwritten")
        if downloaded:
            path.write_bytes(content)
        records.append({"file": name, "rows": len(rows), "encoding": encoding,
                        "sha256": hashlib.sha256(content).hexdigest(), "source": SOURCE,
                        "sequence": seq, "downloaded_this_run": downloaded})
        print(f"{name}: {len(rows)} rows ({encoding})")
    manifest = {"checked_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "license": "공공누리 1유형 (출처표시)", "files": records}
    (folder / "ml_download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
