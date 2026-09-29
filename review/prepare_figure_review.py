"""Prepare the inputs of the figure review from the manuscript: copy the figures in paper order to review/figures/figureN.png
(main text) and efigureN.png (supplement), and write review/figures/captions.md from the captions in the LaTeX sources.

Usage: python prepare_figure_review.py [folder]     (the folder, inside review/, defaults to figures; a second folder lets one
                                                     review run while another is still reading the first)
"""
import pathlib, re, shutil, sys
HERE = pathlib.Path(__file__).parent; ROOT = HERE.parent
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else "figures"); OUT.mkdir(exist_ok=True)


def braces(s, i):
    """The text inside the braces that open at s[i] == '{', and the index after the closing brace."""
    assert s[i] == "{"; depth = 0
    for j in range(i, len(s)):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0:
            return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def figures(tex):
    out = []
    for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, flags=re.S):
        body = m.group(1)
        g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body); c = body.find("\\caption"); lab = re.search(r"\\label\{(fig:[^}]*)\}", body)
        if not (g and c >= 0 and lab):
            continue
        cap, _ = braces(body, body.index("{", c))
        out.append((g.group(1), lab.group(1), re.sub(r"^\\unboldmath\s*", "", cap.strip())))
    return out


def main():
    for p in OUT.glob("*figure*.png"):
        p.unlink()
    lines = []
    for src, prefix in ((ROOT / "paper" / "main.tex", "figure"), (ROOT / "paper" / "supplement.tex", "efigure")):
        for k, (png, label, cap) in enumerate(figures(src.read_text()), start=1):
            name = f"{prefix}{k}.png"
            src_png = pathlib.Path(png).with_suffix(".png").name       # the paper includes the PDF; the reviewers read its PNG twin
            shutil.copyfile(ROOT / "figures" / src_png, OUT / name)
            lines.append(f"## {name} ({label})\n\n{cap}\n")
            print(f"{name:14s} <- {src_png:30s} {label}")
    (OUT / "captions.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
