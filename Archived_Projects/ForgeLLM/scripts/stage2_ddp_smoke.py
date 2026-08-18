"""Run ForgeLLM's two-process CPU DDP correctness gate."""

from forgellm.model.distributed import run_cpu_ddp_smoke


def main() -> None:
    """Run the smoke and report success only after both workers exit cleanly."""
    run_cpu_ddp_smoke()
    print("G2-Systems DDP smoke passed: two Gloo replicas share identical gradients.")


if __name__ == "__main__":
    main()
