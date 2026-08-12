#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import statistics as st


def main(path):
    from comet import download_model, load_from_checkpoint

    try:
        import torch
        gpus = 1 if torch.cuda.is_available() else 0
    except Exception:
        gpus = 0
    print("scoring %s on %s" % (path, "GPU (cuda)" if gpus else "CPU"))

    model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))

    rows = json.load(open(path, encoding="utf-8"))
    ok = [r for r in rows if r.get("status") == "ok" and r.get("translation", "").strip()]
    data = [{"src": r["prompt_en"], "mt": r["translation"]} for r in ok]
    scores = model.predict(data, batch_size=8, gpus=gpus).scores

    for r, s in zip(ok, scores):
        r["cometkiwi"] = float(s)

    # write scores back in place (keeps refused/error rows untouched, no re-ordering)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    n = len(scores)
    ssorted = sorted(scores)
    p10 = ssorted[int(0.1 * n)]
    print("n=%d mean=%.3f median=%.3f p10=%.3f min=%.3f" % (
        n, st.mean(scores), st.median(scores), p10, min(scores)))

    print("\nbottom decile (review these — see §8a for the automatic retry rule):")
    scored = sorted(((r["cometkiwi"], r["itemnum"]) for r in ok))
    for score, itemnum in scored[:max(1, n // 10)]:
        flag = "  < 0.70, goes to §8a" if score < 0.70 else ""
        print("  item %3d : %.3f%s" % (itemnum, score, flag))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/harmful_hi.json")
