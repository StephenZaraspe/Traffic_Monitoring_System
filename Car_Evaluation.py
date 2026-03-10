"""
Counting Error Rate (CER) Evaluation Script
=============================================
Gemini's Point 3: mAP is a detection metric. Your thesis is about COUNTING.
This script compares your tracker's output against a manual human ground truth
and produces the CER metric your panel will actually care about.

Formula:
    CER (%) = |predicted_count - ground_truth_count| / ground_truth_count × 100

A CER of 0% = perfect counting accuracy.
A CER of 10% = off by 10% from the true count.

HOW TO USE:
  1. Watch your tracked output video manually
  2. Count every vehicle that crosses the counting line yourself (ground truth)
  3. Enter your counts in GROUND_TRUTH below
  4. Run: python cer_evaluation.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Enter your manual human counts here
# Watch the video and count every vehicle crossing the line in each direction
# ─────────────────────────────────────────────────────────────────────────────
GROUND_TRUTH = {
    # Format: 'class_name': {'in': count, 'out': count}
    # Replace 0s with your actual manual counts
    'bus':        {'in': 0,  'out': 0},
    'cars':       {'in': 0,  'out': 0},
    'e-jeepney':  {'in': 0,  'out': 0},
    'jeepney':    {'in': 0,  'out': 0},
    'motorcycle': {'in': 0,  'out': 0},
    'trike':      {'in': 0,  'out': 0},
    'trucks':     {'in': 0,  'out': 0},
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Enter your tracker's output counts here
# Copy from the terminal output after running traffic_tracker_fixed.py
# ─────────────────────────────────────────────────────────────────────────────
TRACKER_OUTPUT = {
    'bus':        {'in': 0,  'out': 0},
    'cars':       {'in': 59, 'out': 36},
    'e-jeepney':  {'in': 0,  'out': 0},
    'jeepney':    {'in': 0,  'out': 0},
    'motorcycle': {'in': 51, 'out': 22},
    'trike':      {'in': 0,  'out': 2},
    'trucks':     {'in': 5,  'out': 1},
}


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ENGINE — no need to edit below this line
# ─────────────────────────────────────────────────────────────────────────────

def compute_cer(predicted, ground_truth):
    """
    Counting Error Rate:
        CER = |P - GT| / GT × 100
    Returns None if ground_truth is 0 (undefined).
    """
    if ground_truth == 0:
        return None  # can't compute CER with zero ground truth
    return abs(predicted - ground_truth) / ground_truth * 100


def compute_counting_accuracy(predicted, ground_truth):
    """
    Counting Accuracy (complement of CER):
        CA = max(0, 1 - |P - GT| / GT) × 100
    """
    if ground_truth == 0:
        return None
    return max(0.0, 1.0 - abs(predicted - ground_truth) / ground_truth) * 100


def run_cer_evaluation():
    CLASS_NAMES = ['bus', 'cars', 'e-jeepney', 'jeepney', 'motorcycle', 'trike', 'trucks']

    print("=" * 70)
    print("  COUNTING ERROR RATE (CER) EVALUATION")
    print("  Metric: CER(%) = |Predicted - GroundTruth| / GroundTruth × 100")
    print("=" * 70)

    # Check if ground truth is filled in
    all_zero = all(
        v['in'] == 0 and v['out'] == 0
        for v in GROUND_TRUTH.values()
    )
    if all_zero:
        print("\n  ⚠ WARNING: All ground truth values are 0.")
        print("  Please watch your video and fill in GROUND_TRUTH manually.")
        print("  Then re-run this script.\n")
        return

    print(f"\n{'Class':<14} {'GT_in':>6} {'GT_out':>7} {'PR_in':>6} {'PR_out':>7} "
          f"{'CER_in':>8} {'CER_out':>9} {'CER_total':>10}")
    print("-" * 75)

    total_gt        = 0
    total_predicted = 0
    weighted_cer    = []
    class_results   = {}

    for cls in CLASS_NAMES:
        gt  = GROUND_TRUTH.get(cls,   {'in': 0, 'out': 0})
        pr  = TRACKER_OUTPUT.get(cls, {'in': 0, 'out': 0})

        gt_total = gt['in'] + gt['out']
        pr_total = pr['in'] + pr['out']

        cer_in    = compute_cer(pr['in'],  gt['in'])
        cer_out   = compute_cer(pr['out'], gt['out'])
        cer_total = compute_cer(pr_total,  gt_total)

        total_gt        += gt_total
        total_predicted += pr_total

        if cer_total is not None:
            weighted_cer.append((cer_total, gt_total))

        class_results[cls] = {
            'gt': gt_total, 'pr': pr_total,
            'cer_in': cer_in, 'cer_out': cer_out, 'cer_total': cer_total
        }

        def fmt_cer(v):
            return f"{v:.1f}%" if v is not None else "  N/A"

        print(f"  {cls:<12} {gt['in']:>6} {gt['out']:>7} {pr['in']:>6} {pr['out']:>7} "
              f"{fmt_cer(cer_in):>8} {fmt_cer(cer_out):>9} {fmt_cer(cer_total):>10}")

    # Overall CER (weighted by ground truth count)
    print("-" * 75)
    overall_cer = compute_cer(total_predicted, total_gt)
    overall_ca  = compute_counting_accuracy(total_predicted, total_gt)

    print(f"  {'TOTAL':<12} {total_gt:>6} {'':>7} {total_predicted:>6}")
    print()
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Overall CER          : {overall_cer:.2f}%{' (lower is better)':>28}│" if overall_cer is not None else f"  │  Overall CER          : N/A (no ground truth)         │")
    print(f"  │  Counting Accuracy    : {overall_ca:.2f}%{' (higher is better)':>27}│" if overall_ca is not None else f"  │  Counting Accuracy    : N/A                            │")
    print(f"  └─────────────────────────────────────────────────┘")

    # Per-class accuracy breakdown
    print(f"\n  Per-class Counting Accuracy:")
    for cls in CLASS_NAMES:
        r  = class_results[cls]
        ca = compute_counting_accuracy(r['pr'], r['gt'])
        if ca is not None:
            bar   = "█" * int(ca // 5)
            grade = "✓ Excellent" if ca >= 90 else "~ Good" if ca >= 75 else "⚠ Needs work" if ca >= 50 else "✗ Poor"
            print(f"    {cls:<14}: {ca:>6.1f}%  {bar:<20} {grade}")
        else:
            print(f"    {cls:<14}: N/A (no ground truth provided)")

    # Thesis table output
    print("\n" + "="*70)
    print("  THESIS TABLE — Copy this into your manuscript:")
    print("="*70)
    print(f"  Model evaluated against {total_gt} manually counted vehicles")
    print(f"  over a 2-minute MMDA intersection recording.\n")
    print(f"  {'Class':<14} {'Ground Truth':>13} {'Predicted':>10} {'CER (%)':>9} {'Accuracy':>10}")
    print(f"  {'-'*60}")
    for cls in CLASS_NAMES:
        r   = class_results[cls]
        ca  = compute_counting_accuracy(r['pr'], r['gt'])
        cer = r['cer_total']
        print(f"  {cls:<14} {r['gt']:>13} {r['pr']:>10} "
              f"{f'{cer:.1f}' if cer is not None else 'N/A':>9} "
              f"{f'{ca:.1f}%' if ca is not None else 'N/A':>10}")
    print(f"  {'-'*60}")
    print(f"  {'OVERALL':<14} {total_gt:>13} {total_predicted:>10} "
          f"{f'{overall_cer:.1f}' if overall_cer is not None else 'N/A':>9} "
          f"{f'{overall_ca:.1f}%' if overall_ca is not None else 'N/A':>10}")
    print("="*70)


if __name__ == '__main__':
    run_cer_evaluation()