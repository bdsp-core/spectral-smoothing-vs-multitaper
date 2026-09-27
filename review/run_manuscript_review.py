"""Run bdsp-core/paper-agents-manuscript on this paper through Amazon Bedrock (no code changes to that repo).

Usage (from the review/ directory, so review_config.json is picked up):
    python run_manuscript_review.py main_flat.tex --repo-path .. --iteration-label v3-baseline [--no-scholar] [--agents ...]
"""
import os, sys, runpy, pathlib
import anthropic
AGENTS_REPO = pathlib.Path("/Users/mwestover/GithubRepos/paper-agents-manuscript")
AWS_PROFILE, AWS_REGION, MODEL = "bdsp", "us-east-1", "us.anthropic.claude-opus-5"
sys.path.insert(0, str(AGENTS_REPO))
os.environ.setdefault("ANTHROPIC_API_KEY", "bedrock")          # the pipeline only checks that it is set
import agents                                                    # noqa: E402  (the pipeline re-imports this same module object)


def make_client(api_key=None):
    return anthropic.AnthropicBedrock(aws_profile=AWS_PROFILE, aws_region=AWS_REGION, max_retries=8, timeout=1800.0)


agents.make_client = make_client
argv = sys.argv[1:]
if "--model" not in argv:
    argv += ["--model", MODEL]
if "--output-dir" not in argv and "-o" not in argv:
    argv += ["--output-dir", str(pathlib.Path(__file__).parent / "reports")]
sys.argv = [str(AGENTS_REPO / "run_review.py")] + argv
runpy.run_path(str(AGENTS_REPO / "run_review.py"), run_name="__main__")
