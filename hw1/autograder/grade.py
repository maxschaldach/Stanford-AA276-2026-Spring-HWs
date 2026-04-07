#!/usr/bin/env python3
"""
AA 276 Homework 1 Autograder
Produces /autograder/results/results.json for Gradescope.

Directory layout assumed at runtime (Gradescope standard):
  /autograder/source/       — this file and tests/ live here
  /autograder/submission/   — student-uploaded part1.py and part2.py
  /autograder/results/      — results.json written here
"""

import json
import os
import sys
import pickle
import traceback

SOURCE_DIR     = '/autograder/source'
SUBMISSION_DIR = '/autograder/submission'
RESULTS_FILE   = '/autograder/results/results.json'
TESTS_DIR      = os.path.join(SOURCE_DIR, 'tests')

# Submission must be on the path so `import part1` / `import part2` work.
# Source dir first so autograder utilities take priority over any accidental
# shadowing in the submission.
sys.path.insert(0, SUBMISSION_DIR)
sys.path.insert(0, SOURCE_DIR)

# ── Scoring Configuration ─────────────────────────────────────────────────────
PART1_TESTS = [
    ('state_limits',   3),
    ('control_limits', 3),
    ('safe_mask',      3),
    ('failure_mask',   3),
    ('g',              3),
]  
PART2_TESTS = [
    ('euler_step', 2.5),
    ('roll_out',   2.5),
    ('u_qp',       10),
]  
# ─────────────────────────────────────────────────────────────────────────────


# ── Result checking (mirrors utils/tests.py) ─────────────────────────────────
def _check(result, expected, label=''):
    import torch
    assert type(result) == type(expected), (
        f'Type mismatch{label}: got {type(result).__name__}, '
        f'expected {type(expected).__name__}'
    )
    if isinstance(result, torch.Tensor):
        assert result.dtype == expected.dtype, (
            f'dtype mismatch{label}: got {result.dtype}, expected {expected.dtype}'
        )
        assert torch.allclose(result, expected, rtol=1e-4), (
            f'Value mismatch{label} (not allclose with rtol=1e-4)'
        )
    elif isinstance(result, (int, bool)):
        assert result == expected, (
            f'Value mismatch{label}: got {result!r}, expected {expected!r}'
        )
    elif isinstance(result, (tuple, list)):
        assert len(result) == len(expected), (
            f'Length mismatch{label}: got {len(result)}, expected {len(expected)}'
        )
        for j in range(len(result)):
            _check(result[j], expected[j], label=f'[{j}]')
    else:
        raise NotImplementedError(
            f'Unsupported result type{label}: {type(result).__name__}'
        )


def grade_function(func, test_cases, max_score):
    """
    Run all test cases for one function.
    Returns (earned_score, max_score, output_string).
    Partial credit: each test case is worth max_score / n_cases.
    """
    n = len(test_cases)
    passed = 0
    lines = []
    for i, case in enumerate(test_cases):
        try:
            result = func(*case['args'], **case.get('kwargs', {}))
            _check(result, case['expected'])
            passed += 1
            lines.append(f'  Test {i}: PASSED')
        except AssertionError as e:
            lines.append(f'  Test {i}: FAILED — {e}')
        except Exception as e:
            lines.append(f'  Test {i}: ERROR — {type(e).__name__}: {e}')

    score = max_score * (passed / n) if n > 0 else 0.0
    summary = f'Passed {passed}/{n} test cases.'
    return score, max_score, summary + '\n' + '\n'.join(lines)


def load_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ── u_fn used by roll_out tests (mirrors check2.py) ──────────────────────────
def _make_u_fn():
    import torch
    def u_fn(x):
        u = torch.zeros((len(x), 4))
        u[:, 0] = 9.8
        u[:, 1] = 1
        u[:, 2] = 1
        u[:, 3] = 1
        return u
    return u_fn


def _patch_roll_out_cases(test_cases):
    """Replace the stored u_fn placeholder (index 1) with the real callable."""
    u_fn = _make_u_fn()
    patched = []
    for case in test_cases:
        args = tuple(
            u_fn if i == 1 else case['args'][i]
            for i in range(len(case['args']))
        )
        patched.append({**case, 'args': args})
    return patched


# ── Main grading logic ────────────────────────────────────────────────────────
def grade_part(module_name, func_specs, part_label):
    """Import a module and grade all functions in func_specs.
    Returns a list of Gradescope test-result dicts."""
    results = []

    try:
        mod = __import__(module_name)
    except Exception as e:
        total_max = sum(pts for _, pts in func_specs)
        results.append({
            'name': f'{part_label} — import error',
            'score': 0,
            'max_score': total_max,
            'output': (
                f'Could not import {module_name}.py.\n'
                f'{type(e).__name__}: {e}\n\n'
                f'{traceback.format_exc()}'
            ),
            'visibility': 'visible',
        })
        return results

    for fname, max_score in func_specs:
        pickle_path = os.path.join(TESTS_DIR, module_name, f'{fname}_test_cases.pickle')

        # Try to get the function from the module
        func = getattr(mod, fname, None)
        if func is None:
            results.append({
                'name': f'{part_label} — {fname}',
                'score': 0,
                'max_score': max_score,
                'output': f'ERROR: {fname} not found in {module_name}.py.',
                'visibility': 'visible',
            })
            continue

        # Load test cases
        try:
            test_cases = load_pickle(pickle_path)
        except FileNotFoundError:
            results.append({
                'name': f'{part_label} — {fname}',
                'score': 0,
                'max_score': max_score,
                'output': f'ERROR: Test file not found: {pickle_path}',
                'visibility': 'visible',
            })
            continue
        except Exception as e:
            results.append({
                'name': f'{part_label} — {fname}',
                'score': 0,
                'max_score': max_score,
                'output': f'ERROR loading test cases: {type(e).__name__}: {e}',
                'visibility': 'visible',
            })
            continue

        # Patch roll_out test cases to inject a real u_fn callable
        if fname == 'roll_out':
            test_cases = _patch_roll_out_cases(test_cases)

        # Run the tests
        try:
            score, max_out, output = grade_function(func, test_cases, max_score)
        except Exception as e:
            score, max_out, output = 0, max_score, (
                f'Unexpected error while grading {fname}:\n'
                f'{type(e).__name__}: {e}\n\n{traceback.format_exc()}'
            )

        results.append({
            'name': f'{part_label} — {fname}',
            'score': round(score, 2),
            'max_score': max_out,
            'output': output,
            'visibility': 'visible',
        })

    return results


def main():
    all_tests = []
    all_tests += grade_part('part1', PART1_TESTS, 'Part 1')
    all_tests += grade_part('part2', PART2_TESTS, 'Part 2')

    total_score = sum(t['score'] for t in all_tests)
    total_max   = sum(t['max_score'] for t in all_tests)

    payload = {
        'score':  round(total_score, 2),
        'output': f'Total: {total_score:.1f} / {total_max:.1f}',
        'tests':  all_tests,
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as fh:
        json.dump(payload, fh, indent=2)

    # Echo a summary to stdout (visible in Gradescope's build log)
    print(f'\nAutograder complete. Score: {total_score:.1f} / {total_max:.1f}\n')
    for t in all_tests:
        status = 'PASS' if t['score'] == t['max_score'] else (
            'PARTIAL' if t['score'] > 0 else 'FAIL'
        )
        print(f"  [{status}] {t['name']}: {t['score']:.1f}/{t['max_score']}")


if __name__ == '__main__':
    main()
