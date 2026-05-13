"""
LoRA adapter lifecycle via HuggingFace Hub.

Training notebook (Notebook B) pushes a new adapter after each GRPO run.
Inference notebook (Notebook A) polls for the latest adapter and hot-swaps it.

HF Hub gives us versioning, rollback (by revision hash), and zero infrastructure.
"""
import logging
from huggingface_hub import HfApi, snapshot_download
from config import HF_ADAPTER_REPO, HF_TOKEN

logger = logging.getLogger(__name__)

api = HfApi(token=HF_TOKEN)


def push_adapter(local_path: str, step: int, commit_message: str | None = None) -> str:
    """
    Push a trained LoRA adapter to HF Hub.
    Returns the commit revision hash (use this to pin the exact adapter version).

    Called from the training notebook after each successful GRPO step.
    """
    msg = commit_message or f"adapter: step {step}"
    result = api.upload_folder(
        folder_path=local_path,
        repo_id=HF_ADAPTER_REPO,
        repo_type="model",
        commit_message=msg,
        # Tag the commit so we can reference it by step number
        create_pr=False,
    )
    revision = result.oid  # the commit SHA
    logger.info(f"Adapter step {step} pushed to {HF_ADAPTER_REPO}@{revision}")
    return revision


def pull_adapter(revision: str, local_dir: str = "./active_adapter") -> str:
    """
    Download a specific adapter revision from HF Hub.
    Returns the local path where it was saved.

    Called by the inference notebook when deploying a new checkpoint.
    """
    path = snapshot_download(
        repo_id=HF_ADAPTER_REPO,
        revision=revision,
        local_dir=local_dir,
        token=HF_TOKEN,
        repo_type="model",
    )
    logger.info(f"Adapter {revision} downloaded to {path}")
    return path


def pull_latest_adapter(local_dir: str = "./active_adapter") -> str:
    """Download the latest (main branch) adapter. Used on cold start."""
    return pull_adapter("main", local_dir)


def list_adapter_commits(limit: int = 10) -> list[dict]:
    """List recent adapter commits — useful for rollback decisions."""
    commits = api.list_repo_commits(
        repo_id=HF_ADAPTER_REPO,
        repo_type="model",
    )
    return [
        {"revision": c.commit_id, "message": c.title, "created_at": c.created_at}
        for c in list(commits)[:limit]
    ]
