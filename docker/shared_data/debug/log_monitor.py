import time
import os
import re
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def parse_state_block(block: str) -> Dict[str, str]:
    """Parses a state block into a dictionary."""
    state = {}
    patterns = {
        "mode": r"Operating mode changed to: (.+)",
        "acq_action": r"Next acquisition action: (.+)",
        "elab_action": r"Next elaboration action: (.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, block)
        if match:
            state[key] = match.group(1).strip()
    return state


def summarize_changes(states: List[Dict[str, str]]) -> None:
    """Prints a summary of state changes."""
    if not states:
        return

    initial_state = {}
    for state in states:
        for key in ["mode", "acq_action", "elab_action"]:
            if key not in initial_state and key in state:
                initial_state[key] = state[key]

    print(
        f"  Initial -> Mode: {initial_state.get('mode', 'N/A')}, "
        f"Acquisition: {initial_state.get('acq_action', 'N/A')}, "
        f"Elaboration: {initial_state.get('elab_action', 'N/A')}"
    )

    last_state = initial_state
    for i, current_state in enumerate(states):
        summary = []
        if "mode" in current_state and current_state.get("mode") != last_state.get(
            "mode"
        ):
            summary.append(f"Mode -> {current_state['mode']}")
            last_state["mode"] = current_state["mode"]

        if "acq_action" in current_state and current_state.get(
            "acq_action"
        ) != last_state.get("acq_action"):
            summary.append(f"Acquisition -> {current_state['acq_action']}")
            last_state["acq_action"] = current_state["acq_action"]

        if "elab_action" in current_state and current_state.get(
            "elab_action"
        ) != last_state.get("elab_action"):
            summary.append(f"Elaboration -> {current_state['elab_action']}")
            last_state["elab_action"] = current_state["elab_action"]

        if summary:
            print(f"  Step {i+1}: " + ", ".join(summary))


def visualize_changes(states: List[Dict[str, str]], command: str, output_dir: str):
    """Visualizes state changes and saves the plot to a file."""
    if not states:
        return

    keys = ["mode", "acq_action", "elab_action"]
    data = {key: [] for key in keys}
    last_values = {}

    for state in states:
        for key in keys:
            if key in state:
                last_values[key] = state[key]
            # Use 'N/A' if a value has not been seen yet
            data[key].append(last_values.get(key, "N/A"))

    steps = range(1, len(states) + 1)
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f'State Changes After Command: "{command}"', fontsize=16)

    titles = ["Operating Mode", "Acquisition Action", "Elaboration Action"]
    for i, key in enumerate(keys):
        ax = axes[i]
        ax.step(steps, data[key], where="post")
        ax.set_title(titles[i])
        ax.set_ylabel("State")
        ax.tick_params(axis="y", labelrotation=0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    axes[-1].set_xlabel("Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    safe_command = re.sub(r"[^a-zA-Z0-9_-]", "_", command)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"plot_{safe_command}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(filepath)
    plt.close(fig)
    print(f"  [+] Visualization saved to '{os.path.basename(filepath)}'")


def monitor_log(log_file: str, command_file: str, poll_interval: float = 1.0):
    """Monitors the log file and summarizes state changes."""
    print(f"Monitoring '{os.path.basename(log_file)}' for changes...")
    last_pos = 0
    script_dir = os.path.dirname(log_file)
    plots_dir = os.path.join(script_dir, "plots")

    known_commands = set()
    if os.path.exists(command_file):
        with open(command_file, "r") as f:
            known_commands = {line.strip() for line in f if line.strip()}

    if not known_commands:
        print(
            f"Warning: Could not load commands from '{os.path.basename(command_file)}'."
        )

    try:
        while True:
            if not os.path.exists(log_file):
                time.sleep(poll_interval)
                continue

            with open(log_file, "r") as f:
                f.seek(last_pos)
                new_content = f.read()
                last_pos = f.tell()

            if new_content:
                chunks = re.split(r"^-{20,}\n", new_content, flags=re.MULTILINE)
                current_states = []
                last_command = "initial_states"
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    if chunk in known_commands:
                        if current_states:
                            summarize_changes(current_states)
                            visualize_changes(current_states, last_command, plots_dir)
                            current_states = []
                        print(f"\n[COMMAND]: {chunk}")
                        last_command = chunk
                    else:
                        state = parse_state_block(chunk)
                        if state:
                            current_states.append(state)

                if current_states:
                    summarize_changes(current_states)
                    visualize_changes(current_states, last_command, plots_dir)

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except ImportError:
        print(
            "\nError: `matplotlib` is not installed. Please run 'pip install matplotlib' to enable visualization."
        )


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    debug_log_path = os.path.join(script_dir, "debug_log.txt")
    commands_path = os.path.join(script_dir, "counting_backwards.txt")
    monitor_log(debug_log_path, commands_path)
