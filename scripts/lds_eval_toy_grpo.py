# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "scipy",
# ]
# ///

from __future__ import annotations

import argparse
import time
import copy
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from influence_rlvr.toy_grpo import (
    ToyGRPOExample,
    ToyRolloutMode,
    clone_toy_model,
    compute_toy_fisher_influence,
    compute_toy_historical_fisher_influence,
    exact_expected_reward,
    train_toy_grpo,
    reward_for_sequences,
    _ALL_TWO_TOKEN_SEQUENCES,
    GradientObjective,
    _toy_objective_and_debug,
    ToyHistoricalWeightMode
)


class ToyAutoregressiveMLP(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 16, output_dim: int = 2, use_bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.first = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=use_bias)
        )
        self.second = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=use_bias)
        )

    def first_token_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.first(z)

    def second_token_logits(
        self,
        z: torch.Tensor,
        first_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if first_tokens.ndim == 1:
            first_tokens = first_tokens.unsqueeze(1)
        aug = torch.cat([z, first_tokens.to(dtype=z.dtype)], dim=1)
        return self.second(aug)

    def per_token_log_probs(
        self,
        z: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(dtype=torch.float32)
        n_seq = int(sequences.shape[0])
        z_rep = z.expand(n_seq, -1)
        first_logits = self.first_token_logits(z_rep)
        first_log_probs = F.log_softmax(first_logits, dim=-1)
        first_tokens = sequences[:, 0]
        first_lp = first_log_probs.gather(1, first_tokens.unsqueeze(1)).squeeze(1)

        second_logits = self.second_token_logits(z_rep, first_tokens.float())
        second_log_probs = F.log_softmax(second_logits, dim=-1)
        second_tokens = sequences[:, 1]
        second_lp = second_log_probs.gather(1, second_tokens.unsqueeze(1)).squeeze(1)

        return torch.stack([first_lp, second_lp], dim=1)

    def sequence_log_probs(
        self,
        z: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        return self.per_token_log_probs(z, sequences).sum(dim=1)

    def exact_sequence_distribution(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = _ALL_TWO_TOKEN_SEQUENCES.to(self.device)
        sequence_log_probs = self.sequence_log_probs(z.to(self.device), sequences)
        return sequences, sequence_log_probs.exp()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def generate_dataset(n: int = 1000, input_dim: int = 20, seed: int = 42) -> list[ToyGRPOExample]:
    torch.manual_seed(seed)
    z = torch.randn(n, input_dim)
    # Rule: sum(z1, z2, z3) > 0 -> [1, 0], else [0, 1]
    rule_sum = z[:, 0] + z[:, 1] + z[:, 2]
    targets = torch.zeros(n, 2, dtype=torch.long)
    targets[rule_sum > 0] = torch.tensor([1, 0])
    targets[rule_sum <= 0] = torch.tensor([0, 1])
    
    examples = []
    for i in range(n):
        examples.append(ToyGRPOExample(
            name=f"ex_{i}",
            z=tuple(z[i].tolist()),
            target=tuple(targets[i].tolist()),
        ))
    return examples


def train_model_with_history(
    dataset: Sequence[ToyGRPOExample],
    steps: int = 1000,
    lr: float = 1e-3, 
    hidden_dim: int = 16,
    seed: int = 0,
    save_checkpoints: bool = True,
    use_adam: bool = True
) -> dict:
    torch.manual_seed(seed)
    model = ToyAutoregressiveMLP(hidden_dim=hidden_dim)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    checkpoints = {}
    if save_checkpoints:
        checkpoints[0] = copy.deepcopy(model.state_dict())
        
    history = []
    for step in range(1, steps + 1):
        example = dataset[(step - 1) % len(dataset)]
        old_model = clone_toy_model(model)
        optimizer.zero_grad()
        
        objective, _, debug = _toy_objective_and_debug(
            model,
            example,
            G=4,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            seed=seed + step,
            epsilon=0.2,
            beta=0.0,
            old_model=old_model, 
            ref_model=None,
            advantage_eps=1e-4,
            objective_mode=GradientObjective.GRPO_TRAIN
        )
        objective.backward()
        optimizer.step()
        
        history.append({
            "step": step,
            "example_name": example.name,
            "loss": float(objective.item())
        })
        
        if save_checkpoints:
            checkpoints[step] = copy.deepcopy(model.state_dict())
            
    return {
        "model": model,
        "history": history,
        "checkpoints": checkpoints
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=200) 
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--m-subsets", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000, help="Total GRPO steps for full model")
    parser.add_argument("--subset-steps", type=int, default=1000, help="GRPO steps for subset models")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-damp", type=float, default=1.0)
    parser.add_argument("--no-adam", action="store_true")
    parser.add_argument(
        "--historical-weight-mode",
        choices=[mode.value for mode in ToyHistoricalWeightMode],
        default=ToyHistoricalWeightMode.ALL_SAMPLES.value,
        help=(
            "How historical trajectory Fisher weights train examples at each step: "
            "`active_only` keeps only the example actually used at that step; "
            "`all_samples` includes every train example at every step."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/lds_toy_grpo", help="Directory to save results.")
    args = parser.parse_args()

    use_adam = not args.no_adam
    hist_weight_mode = ToyHistoricalWeightMode.parse(args.historical_weight_mode)
    
    # Setup output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    # Save config
    config = vars(args)
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Generating dataset N={args.n_train}...")
    all_examples = generate_dataset(n=args.n_train + 200, seed=args.seed)
    train_examples = all_examples[:args.n_train]
    
    hard_test_examples = []
    for ex in all_examples[args.n_train:]:
        val = abs(ex.z[0] + ex.z[1] + ex.z[2])
        if val < 0.3:
            hard_test_examples.append(ex)
        if len(hard_test_examples) >= args.n_test:
            break
    test_examples = hard_test_examples
    print(f"Selected {len(test_examples)} hard test examples.")

    print(f"Training full model with history ({'Adam' if use_adam else 'SGD'}) for {args.steps} steps...")
    train_result = train_model_with_history(
        train_examples, 
        steps=args.steps, 
        lr=args.lr, 
        hidden_dim=args.hidden_dim, 
        seed=args.seed,
        use_adam=use_adam
    )
    full_model = train_result["model"]
    
    # Accuracy check
    correct = 0
    for ex in train_examples:
        if exact_expected_reward(full_model, ex) > 0.5:
            correct += 1
    accuracy = 100 * correct / args.n_train
    print(f"Full model training accuracy: {correct}/{args.n_train} ({accuracy:.1f}%)")

    print(f"Computing Historical Trajectory Influence Functions (mode={hist_weight_mode.value})...")
    test_ifs = []
    for i, test_ex in enumerate(test_examples):
        hist_inf = compute_toy_historical_fisher_influence(
            full_model,
            checkpoints=train_result["checkpoints"],
            train_history=train_result["history"],
            train_examples=train_examples,
            test_example=test_ex,
            learning_rate=args.lr,
            lambda_damp=args.lambda_damp,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            historical_weight_mode=hist_weight_mode
        )
        scores = hist_inf["repo_scores"]
        print(f"  Test Example {i} Trajectory IF stats: Mean={scores.mean():.4f}, Std={scores.std():.4f}, Max={scores.max():.4f}")
        test_ifs.append(scores)

    print(f"Training {args.m_subsets} subset models for LDS verification...")
    actual_rewards = np.zeros((args.n_test, args.m_subsets))
    predicted_rewards = np.zeros((args.n_test, args.m_subsets))

    torch.manual_seed(args.seed + 1)
    subset_masks = torch.randint(0, 2, (args.m_subsets, args.n_train), dtype=torch.bool)

    start_time = time.time()
    for j in range(args.m_subsets):
        if j % 10 == 0 and j > 0:
            elapsed = time.time() - start_time
            print(f"  Subset {j}/{args.m_subsets} (Elapsed: {elapsed:.1f}s)...")
        
        mask = subset_masks[j]
        subset_train = [train_examples[i] for i in range(args.n_train) if mask[i]]
        if not subset_train: continue
            
        sub_res = train_model_with_history(
            subset_train, 
            steps=args.subset_steps, 
            lr=args.lr, 
            hidden_dim=args.hidden_dim, 
            seed=args.seed + j + 100,
            save_checkpoints=False,
            use_adam=use_adam
        )
        sub_model = sub_res["model"]
        
        for i in range(args.n_test):
            actual_rewards[i, j] = exact_expected_reward(sub_model, test_examples[i])
            predicted_rewards[i, j] = test_ifs[i][mask.numpy()].sum()

    print("\nLDS Evaluation Results for Trajectory Influence:")
    corrs = []
    test_results = []
    for i in range(args.n_test):
        actual_std = np.std(actual_rewards[i])
        pred_std = np.std(predicted_rewards[i])
        
        result_entry = {
            "test_idx": i,
            "actual_std": float(actual_std),
            "pred_std": float(pred_std),
        }
        
        if actual_std == 0 or pred_std == 0:
            print(f"  Test Example {i}: Undefined variance (Actual Std={actual_std:.4f}, Pred Std={pred_std:.4f}).")
            result_entry["correlation"] = None
        else:
            corr, p = spearmanr(actual_rewards[i], predicted_rewards[i])
            corrs.append(corr)
            result_entry["correlation"] = float(corr)
            result_entry["p_value"] = float(p)
            print(f"  Test Example {i}: Corr={corr:.4f}, p={p:.4e}")
        
        test_results.append(result_entry)

    avg_corr = np.mean(corrs) if corrs else 0.0
    if corrs:
        print(f"\nAverage Trajectory LDS Correlation: {avg_corr:.4f}")

    # Save quantitative results
    final_results = {
        "accuracy": accuracy,
        "average_correlation": float(avg_corr),
        "test_results": test_results,
        "total_time": time.time() - start_time
    }
    with (run_dir / "results.json").open("w") as f:
        json.dump(final_results, f, indent=2)

    # Save qualitative analysis
    qual_path = run_dir / "qualitative_analysis.txt"
    with qual_path.open("w") as f:
        if corrs:
            for i in range(args.n_test):
                f.write("\n" + "="*50 + "\n")
                f.write(f"QUALITATIVE ANALYSIS: Test Example {i}\n")
                target_ex = test_examples[i]
                target_sum = sum(target_ex.z[:3])
                f.write(f"Test Input z1+z2+z3: {target_sum:.4f}\n")
                f.write(f"Test Target: {target_ex.target}\n")
                f.write("="*50 + "\n")
                
                scores = test_ifs[i]
                sorted_indices = np.argsort(scores)
                
                f.write("\nTop 5 HARMFUL Examples (Lowest IF):\n")
                for idx in sorted_indices[:5]:
                    ex = train_examples[idx]
                    rule_sum = sum(ex.z[:3])
                    f.write(f"  {ex.name}: Score={scores[idx]:.4f}, z1+z2+z3={rule_sum:.4f}, Target={ex.target}\n")
                    
                f.write("\nTop 5 HELPFUL Examples (Highest IF):\n")
                for idx in sorted_indices[-5:][::-1]:
                    ex = train_examples[idx]
                    rule_sum = sum(ex.z[:3])
                    f.write(f"  {ex.name}: Score={scores[idx]:.4f}, z1+z2+z3={rule_sum:.4f}, Target={ex.target}\n")
            
            print(f"\nQualitative analysis saved to {qual_path}")
            # Print Test Example 0 to console as well
            print("\n" + "="*50)
            print(f"QUALITATIVE ANALYSIS: Test Example 0")
            target_ex = test_examples[0]
            target_sum = sum(target_ex.z[:3])
            print(f"Test Input z1+z2+z3: {target_sum:.4f}")
            print(f"Test Target: {target_ex.target}")
            print("="*50)
            
            scores = test_ifs[0]
            sorted_indices = np.argsort(scores)
            print("\nTop 5 HELPFUL Examples (Highest IF):")
            for idx in sorted_indices[-5:][::-1]:
                ex = train_examples[idx]
                rule_sum = sum(ex.z[:3])
                print(f"  {ex.name}: Score={scores[idx]:.4f}, z1+z2+z3={rule_sum:.4f}, Target={ex.target}")
        else:
            f.write("\nNo valid correlations computed.")

    # Save raw rewards for plotting
    with (run_dir / "rewards.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["subset_idx"]
        for i in range(args.n_test):
            header.extend([f"actual_reward_{i}", f"predicted_reward_{i}"])
        writer.writerow(header)
        for j in range(args.m_subsets):
            row = [j]
            for i in range(args.n_test):
                row.extend([actual_rewards[i, j], predicted_rewards[i, j]])
            writer.writerow(row)
    print(f"Raw rewards saved to {run_dir / 'rewards.csv'}")

    # Generate and save plots
    if corrs:
        fig, axes = plt.subplots(1, args.n_test, figsize=(5 * args.n_test, 5), squeeze=False)
        for i in range(args.n_test):
            ax = axes[0, i]
            ax.scatter(predicted_rewards[i], actual_rewards[i], alpha=0.5)
            ax.set_xlabel("Predicted Reward (IF Sum)")
            ax.set_ylabel("Actual Reward (Subset Model)")
            
            # Clean correlation label for title
            corr_val = next((res["correlation"] for res in test_results if res["test_idx"] == i), 0.0)
            ax.set_title(f"Test Example {i}\nSpearman Corr: {corr_val:.4f}")
            ax.grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plot_path = run_dir / "lds_correlation.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Correlation plots saved to {plot_path}")


if __name__ == "__main__":
    main()
