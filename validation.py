import os
import re
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Validate GutBrainIE CLEF2025 submission folder structure"
    )
    parser.add_argument(
        "base_path",
        help="Path to the base directory containing run subfolders"
    )
    args = parser.parse_args()
    base_path = args.base_path

    if not os.path.isdir(base_path):
        print(f"Error: '{base_path}' is not a valid directory.")
        sys.exit(1)

    # List all immediate subdirectories
    subdirs = [d for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d))]

    # Regex for TeamID_TaskID_RunID_SystemDesc
    pattern = re.compile(r'^([^_]+)_(T6(?:21|22|23|1))_([^_]+)(?:_(.+))?$')

    parsed = []
    errors = []

    for d in subdirs:
        m = pattern.match(d)
        if not m:
            errors.append(f"Folder '{d}' does not follow '<TeamID>_<TaskID>_<RunID>[_<SystemDesc>]' pattern.")
        else:
            team, task, run_id, system = m.group(1), m.group(2), m.group(3), m.group(4)
            parsed.append({
                "folder": d,
                "team": team,
                "task": task,
                "run": run_id,
                "system": system or None
            })

    if errors:
        for e in errors:
            print("Error:", e)
        sys.exit(1)

    if not parsed:
        print("Error: No subfolders matching the required naming convention were found.")
        sys.exit(1)

    # Ensure single TeamID across all
    teams = {p["team"] for p in parsed}
    if len(teams) > 1:
        print("Error: Multiple TeamIDs found:", ", ".join(sorted(teams)))
        sys.exit(1)
    team = teams.pop()

    # Count tasks
    task_counts = {}
    for p in parsed:
        task_counts[p["task"]] = task_counts.get(p["task"], 0) + 1

    # Collect runIDs and system descriptions
    runs = {p["run"] for p in parsed}
    systems = {s for s in (p["system"] for p in parsed) if s}

    # Print summary
    print(f"Team ID: {team}")
    print("Tasks:")
    for t in sorted(task_counts.keys(), key=lambda x: int(x.replace('T6', ''))):
        count = task_counts[t]
        print(f"  {t}: {count} folder{'s' if count != 1 else ''}")
    print("Runs:", ", ".join(sorted(runs)))
    print("System Descriptions:", ", ".join(sorted(systems)) if systems else "None")

    # Verify presence of JSON and META files in each
    all_ok = True
    for p in parsed:
        folder_path = os.path.join(base_path, p["folder"])
        expected_json = p["folder"] + ".json"
        expected_meta = p["folder"] + ".meta"
        actual_files = set(os.listdir(folder_path))
        missing = [f for f in (expected_json, expected_meta) if f not in actual_files]
        if missing:
            print(f"Error in '{p['folder']}': missing file(s): {', '.join(missing)}")
            all_ok = False

    if not all_ok:
        sys.exit(1)

    print("All folders and files are valid.")

if __name__ == "__main__":
    main()
