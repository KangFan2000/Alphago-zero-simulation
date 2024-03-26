"""
Here is the Hex code from COMP34111. This is only used for run one game with agents to have a look.
"""
import shlex
import subprocess
from sys import argv, platform
from os.path import realpath, sep


def extract_agents(arguments):
    """Returns two lists with agents separated from other arguments.
    Ignores first argument because that is the name of the script.

    Any badly formatted agents will be printed and ignored.
    """

    agents = []
    other_args = []
    for argument in arguments:
        if ("a=" in argument or "-agent" in argument):
            try:
                name, cmd = argument.split("=")[1].split(";")
                agents.append(f'"{argument}"')
            except Exception:
                print(f"Agent '{argument}' is not in correct format.")
        else:
            other_args.append(argument)
    return (agents, other_args[1:])


def get_main_cmd():
    """Checks the OS to specify python or python3 and creates a relative path
    command to src/main.py.
    """

    main_cmd = "python"
    if (platform != "win32"):
        main_cmd += "3"

    main_path = sep.join(realpath(__file__).split(sep)[:-1])
    main_path += f"{sep}src{sep}main.py"

    main_cmd += " " + main_path
    return main_cmd


def main():
    """Checks that at most two agents are specified and that they
    are unique, then runs the main script with the given args.
    """

    agents, arguments = extract_agents(argv)
    if (len(agents)) > 2:
        print("ERROR: Too many agents specified. Aborted.")
        return
    elif (len(agents) != len(set(agents))):
        print("ERROR: Agent strings must be unique. Aborted.")

    cmd = (
        get_main_cmd() + " " +
        " ".join(arguments) + " " +
        " ".join(agents)
    )
    if (platform != "win32"):
        cmd = shlex.split(cmd)

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
