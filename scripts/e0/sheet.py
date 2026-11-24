import json, cv2, numpy as np
from pathlib import Path
E = Path("/home/<user>/bxjudge/runs/e0-baseline-20261124/truth")
W = Path("/mnt/c/Users/<user>/AppData/Local/Temp/claude/0d05d5ae05dd15db/76cf77f2-3949-4532-8b09-753d801c1ee2/scratchpad/w/kf")
K = json.loads((E / "keyframes.json").read_text())
C, CW, CH = 3, 112, 112
cells = []
for v, d in K.items():
    for i, fr in enumerate(d["frames"]):
        im = cv2.imread(str(E / f"{v}_{fr['frame']:05d}.jpg"))
        cell = np.full((CH + 22, C * CW + 4, 3), 40, np.uint8)
        cv2.putText(cell, f"{v[-6:]}#{i} f{fr['frame']}", (2, 16), 0, 0.45, (0, 255, 255), 1)
        for j, b in enumerate(fr["boxes"][:C]):
            x0, y0, x1, y1 = b; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * 0.75
            crop = im[int(max(0, cy - s)):int(cy + s), int(max(0, cx - s)):int(cx + s)]
            if crop.size == 0: continue
            crop = cv2.resize(crop, (CW - 4, CH - 4))
            cv2.putText(crop, str(j), (3, 18), 0, 0.6, (255, 255, 255), 2)
            cell[22:22 + CH - 4, j * CW + 2:j * CW + CW - 2] = crop
        if len(fr["boxes"]) > C: cv2.putText(cell, f"+{len(fr['boxes'])-C}", (C * CW - 40, 16), 0, 0.45, (0, 0, 255), 1)
        cells.append((v, cell))
cols, rows = 5, 8
per = cols * rows
for k in range(0, len(cells), per):
    chunk = [c for _, c in cells[k:k + per]]
    while len(chunk) < per: chunk.append(np.zeros_like(chunk[0]))
    sheet = np.vstack([np.hstack(chunk[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(W / f"sheet{k // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(cells), "cells", (len(cells) + per - 1) // per, "sheets", sheet.shape)
