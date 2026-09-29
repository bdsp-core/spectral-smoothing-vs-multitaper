"""Run bdsp-core/paper-agents-figures on this paper's figures through Amazon Bedrock, with captions injected.

Usage: python run_figure_review.py [--main-only] [--agents story,caption,...]
"""
import os, sys, re, pathlib
import anthropic
AGENTS_REPO = pathlib.Path("/Users/mwestover/GithubRepos/paper-agents-figures")
HERE = pathlib.Path(__file__).parent
AWS_PROFILE, AWS_REGION, MODEL = "bdsp", "us-east-1", "us.anthropic.claude-opus-5"
os.environ["JOURNAL_CONFIG_PATH"] = str(HERE / "journal_config.json")
os.environ.setdefault("ANTHROPIC_API_KEY", "bedrock")
sys.path.insert(0, str(AGENTS_REPO))
import run_review as rr                                          # noqa: E402

argv = sys.argv[1:]
FIG_DIR = pathlib.Path(argv[argv.index("--fig-dir") + 1]).resolve() if "--fig-dir" in argv else HERE / "figures"
caps = {}
txt = (FIG_DIR / "captions.md").read_text()                       # the captions travel with the figures
for m in re.finditer(r"^## (\S+\.png) \((fig:[a-z0-9]+)\)\n\n(.*?)(?=^## |\Z)", txt, flags=re.S | re.M):
    caps[m.group(1)] = m.group(3).strip()
_orig = rr.build_image_content


def build_image_content(path):
    blocks = _orig(path)
    cap = caps.get(os.path.basename(path))
    if cap:
        blocks.append({"type": "text", "text": f"Draft caption for {os.path.basename(path)} (LaTeX source):\n{cap}"})
    return blocks


rr.build_image_content = build_image_content
rr.REPORT_DIR = str(HERE / "reports")
anthropic.Anthropic = lambda **kw: anthropic.AnthropicBedrock(aws_profile=AWS_PROFILE, aws_region=AWS_REGION, max_retries=8, timeout=1800.0)
if "--fig-dir" not in argv and not any(a.endswith(".png") for a in argv):
    argv += ["--fig-dir", str(FIG_DIR)]
if "--model" not in argv:
    argv += ["--model", MODEL]
sys.argv = [str(AGENTS_REPO / "run_review.py")] + argv
rr.main()
