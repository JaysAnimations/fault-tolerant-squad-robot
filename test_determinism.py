"""
test_determinism.py
===================
Run one seed twice and assert the CSV rows come out byte-identical.

    py test_determinism.py            C3 on seed 7 (~50 s)
    py test_determinism.py C2 42      any condition, any seed

WHY THIS EXISTS. "Same seed, byte-identical result" is one of the project's
strongest claims and until now it was checked by eye -- somebody noticing
that two printed tables agreed. Eyes do not check forty-five columns, and
they do not run at all once the session ends. This makes the claim
auditable in one command.

WHAT IT COMPARES, AND WHY THAT AND NOT THE PRINTED SUMMARY. The two runs
are serialised through the SAME csv.DictWriter over the SAME COLUMNS list
that run_experiments.py writes results.csv with, so what is compared is
literally the text that would land in the file -- not a re-formatted
summary that might round two different numbers to the same display width.

C3 IS THE DEFAULT ON PURPOSE. It is the only condition that runs detection
AND recovery, so it draws on the most RNG: sensor noise, packet loss, fault
timing, the auction, and every quarantine decision. A condition that
consumed fewer random numbers would be an easier test than the suite
actually needs to pass.
"""
import csv
import io
import sys
import time

import run_experiments


def row_text(condition, seed):
    """One mission, serialised exactly as results.csv would hold it."""
    row = run_experiments.run_one(condition, seed, save_trace=False)
    buf = io.StringIO()
    # extrasaction="ignore" drops _wall_s, which is wall-clock and must not
    # be compared -- the same mission legitimately takes a different number
    # of seconds on a busy machine.
    writer = csv.DictWriter(buf, fieldnames=run_experiments.COLUMNS,
                            extrasaction="ignore", lineterminator="\n")
    writer.writerow(row)
    return buf.getvalue()


def main(condition="C3", seed=7):
    print(f"DETERMINISM -- {condition} seed {seed}, run twice\n")

    started = time.time()
    first = row_text(condition, seed)
    print(f"  run 1 done ({time.time() - started:.0f} s)")

    started = time.time()
    second = row_text(condition, seed)
    print(f"  run 2 done ({time.time() - started:.0f} s)")
    print()

    if first == second:
        print(f"  {len(first)} bytes, identical across both runs.")
        print("PASS -- the same seed produced a byte-identical row.")
        return 0

    # Name the columns that moved. "The rows differ" is not a useful
    # failure message when there are forty-five of them.
    a = next(csv.DictReader(io.StringIO(first),
                            fieldnames=run_experiments.COLUMNS))
    b = next(csv.DictReader(io.StringIO(second),
                            fieldnames=run_experiments.COLUMNS))
    print("FAIL -- the same seed produced different results. Columns that moved:")
    for column in run_experiments.COLUMNS:
        if a[column] != b[column]:
            print(f"    {column}: {a[column]!r} -> {b[column]!r}")
    return 1


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args[0] if args else "C3",
                  int(args[1]) if len(args) > 1 else 7))
