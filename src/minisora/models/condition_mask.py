from typing import Dict, Optional
import torch


def build_condition_mask(
    batch_size: int,
    post_patch_frames: int,
    device: torch.device,
    task_probs: Dict[str, float],
) -> Optional[torch.Tensor]:
    if batch_size == 0 or post_patch_frames <= 1:
        return None

    total_prob = sum(task_probs.values())
    if total_prob < 0 or total_prob > 1 + 1e-6:
        raise ValueError(f"Sum of conditioning probabilities must be within [0, 1], got {total_prob}.")
    if total_prob == 0:
        return None

    identity_prob = max(0.0, 1.0 - total_prob)
    is_noisy_mask = torch.ones(batch_size, post_patch_frames, dtype=torch.bool, device=device)
    random_vals = torch.rand(batch_size, device=device)
    has_conditioning = False

    cumulative = identity_prob
    thresholds = [("identity", identity_prob)]
    for name, prob in task_probs.items():
        if prob <= 0:
            continue
        cumulative += prob
        thresholds.append((name, cumulative))

    for sample_idx in range(batch_size):
        draw = random_vals[sample_idx].item()
        selected_mode: Optional[str] = None
        for name, threshold in thresholds:
            if draw < threshold:
                selected_mode = name
                break

        if selected_mode == "identity":
            continue

        if _apply_condition_mode(is_noisy_mask[sample_idx], selected_mode, post_patch_frames):
            has_conditioning = True

    # mask: False = clean, True = noisy
    return is_noisy_mask if has_conditioning else None


def _apply_condition_mode(is_noisy_mask_row: torch.Tensor, mode: str, post_patch_frames: int) -> bool:
    if mode == "continuation":
        # keep an initial context block intact so the model learns to continue videos
        if post_patch_frames < 2:
            return False
        context_frames = torch.randint(1, post_patch_frames, (1,), device=is_noisy_mask_row.device).item()
        is_noisy_mask_row[:context_frames] = False
        return True

    if mode == "random":
        # keep a random subset of frames clean; both the subset and ratio vary
        if post_patch_frames < 2:
            return False
        noisy_ratio = torch.rand(1, device=is_noisy_mask_row.device).item() * 0.8 + 0.2
        is_noisy = torch.rand(post_patch_frames, device=is_noisy_mask_row.device) < noisy_ratio
        if not is_noisy.any():
            # if all frames are clean, set one frame to noisy randomly
            is_noisy[torch.randint(0, post_patch_frames, (1,), device=is_noisy_mask_row.device)] = True
        is_noisy_mask_row[~is_noisy] = False
        return True

    raise ValueError(f"Unknown conditioning mode '{mode}'.")
