from rich.console import Console

def must_parse_cli_int(msg: str,
                       min: int = None,
                       max: int = None,
                       default_value: int = None) -> int:
    console = Console()

    while True:
        string_input = console.input(
            " > " +
            msg +
            f" [  default={default_value} ]:"
        )

        if string_input == '' and default_value is not None:
            sel = default_value
            return sel
        else:
            try:
                sel = int(string_input)
                if (min is not None and max is not None) and (sel < min or sel >= max):
                    console.log(f"{sel} is out of range")
                    continue
                else:
                    return sel
            except Exception as e:
                console.log(f"input {string_input} is invalid")
                continue

must_parse_cli_int("Frame Queue Size",min=0, max=1000, default_value=100)