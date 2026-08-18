"""Docker-isolated code-generation executor for data-analysis benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from repopilot.benchmarks.contracts import (
    ArtifactRef,
    BenchmarkTask,
    EnvironmentHandle,
    PreparedTask,
    SharedRunMetrics,
    TaskOutput,
)
from repopilot.core.contracts import Message, ModelRequest
from repopilot.providers.base import ModelProvider

_PYTHON_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class DockerDataAnalysisConfig:
    image: str = "repopilot-dabench:py311-v1"
    max_attempts: int = 2
    timeout_seconds: int = 120
    max_output_tokens: int = 1_024
    dtype_guardrail: bool = True
    """Inject numeric-coercion guidance (P8C D1). Set False to reproduce the pre-D1 baseline."""
    parse_robustness_guardrail: bool = False
    """Inject parse-robustness guidance (P8C D3). Additive on top of dtype_guardrail; default
    False so D2/base stay byte-for-byte reproducible. Targets the 3 crash-type failures in the
    stable-failure core set (0028/0056/0062): categorical aggregation, bracketed/range strings,
    and silent astype failures."""

    def __post_init__(self) -> None:
        if not self.image.strip() or self.max_attempts <= 0:
            raise ValueError("Docker image and positive attempt count are required")
        if self.timeout_seconds <= 0 or self.max_output_tokens <= 0:
            raise ValueError("Docker and model budgets must be positive")


class DockerDataAnalysisExecutor:
    """Generate Python, execute it without network access, and retry on feedback."""

    def __init__(
        self, provider: ModelProvider, config: DockerDataAnalysisConfig | None = None
    ) -> None:
        self.provider = provider
        self.config = config or DockerDataAnalysisConfig()

    @staticmethod
    def extract_python(content: str) -> str:
        match = _PYTHON_FENCE.search(content)
        code = match.group(1) if match else content
        return code.strip()

    # P8C D1: numeric-coercion guardrail.
    #
    # Failure evidence (run 20260809_p8c_dabench_val35_run3, 5/35 empty answers):
    # the model inspects dtypes correctly, sees an object column, and then writes
    # `series.astype('Int64', errors='ignore')`. pandas silently returns the *original
    # strings* when that cast fails, so the next numeric comparison raises
    # "'>' not supported between instances of 'str' and 'int'" and the task yields no answer.
    # D1 (method-prescriptive) removed 3 of 5 empty answers but broke 3 previously
    # correct tasks: forcing pd.to_numeric(errors='coerce') turned parseable values such as
    # "1 234 (est.)" into NaN, whereas the model's own str.extract(r'(\d+)') had handled them.
    # D2 is goal-prescriptive: state the invariant, forbid the silent-failure API, require a
    # data-loss check, but let the model pick the cleaning method.
    NUMERIC_GUARDRAIL = (
        "Numeric-column rule (goal, not a fixed recipe): before any comparison, sort, or "
        "aggregation, every column you treat as numeric must actually have a numeric dtype. "
        "Never use .astype(..., errors='ignore') - on failure pandas silently keeps the original "
        "strings, which later raises \"'>' not supported between instances of 'str' and 'int'\". "
        "Choose the cleaning method that fits the observed values (str.replace, str.extract, "
        "pd.to_numeric(..., errors='coerce'), ...). After cleaning, verify you did not destroy "
        "data: if the conversion turns more than 20% of the non-null values into NaN, your "
        "method is wrong for this format - switch to a more tolerant extraction instead of "
        "dropping those rows. Never convert a genuinely categorical text column to numbers; "
        "use groupby or pd.get_dummies for those."
    )

    # P8C D3: parse-robustness guardrail (additive, default OFF).
    #
    # Deep diagnosis of the 5-question stable-failure core set (run
    # 20260809_p8c_dabench_val35_run3) showed the set is NOT a uniform capability bottleneck:
    #   - 0028 (hard)  : crash  - .mean() called on categorical 'region' (str) -> TypeError
    #   - 0056 (easy)  : crash  - astype('Int64', errors='ignore') leaves str; '>0' compares
    #                            str vs int -> crash (the classic silent-dtype failure)
    #   - 0062 (medium): crash  - astype(float) on bracketed/range strings like '[123]' ->
    #                            ValueError
    #   - 0055 (easy)  : WRONG  - mean computed but value wrong (reasoning/aggregation error)
    #   - 0109 (hard)  : WRONG  - t-test / means wrong (reasoning error)
    # 3 of 5 are fixable parsing/robustness crashes; 2 are genuine reasoning errors that no
    # prompt change can fix under the frozen model. D3 therefore targets ONLY the 3 crash
    # patterns, with scenario-specific rules derived from the actual stderr traces. It stays
    # goal-oriented (pick the method that fits) but adds the concrete invariants that D2 left
    # implicit and that the model violated in practice.
    PARSE_ROBUSTNESS_GUARDRAIL = (
        "Numeric-cleaning rules (derived from real crash traces in this benchmark):\n"
        "1) Only aggregate columns that are already numeric (int/float) or that you have "
        "explicitly converted with pd.to_numeric(errors='coerce'). Never call .mean()/.sum()/"
        ".quantile()/.std()/.idxmax() on an object/string/category column - it raises TypeError. "
        "Categorical columns (region, sex labels, country names) are for groupby/get_dummies, "
        "not for averaging.\n"
        "2) If a numeric column contains brackets, ranges, or embedded text (e.g. '[123]', "
        "'100-200', '1 234 (est.)'), do NOT call astype(float) on the raw cell - it raises "
        "ValueError. Extract one representative number first, e.g. "
        "df[col].str.extract(r'(\\d+(?:\\.\\d+)?)').iloc[:, 0], or take the lower bound of a "
        "range, then coerce. Verify the result is numeric before comparing/aggregating.\n"
        "3) Never use .astype(..., errors='ignore'): on failure pandas keeps the original "
        "strings and the next numeric comparison crashes. Prefer str.extract (keeps the number, "
        "drops the noise) over blanket to_numeric(errors='coerce') when cells mix numbers and "
        "text; coerce only when the whole cell is numeric after cleaning.\n"
        "After cleaning, if more than 20% of non-null values became NaN, your method is wrong "
        "for this format - switch to a more tolerant extraction. The goal is a correct numeric "
        "column, not a specific API."
    )

    @staticmethod
    def diagnostic_hint(stderr: str) -> str:
        if "not supported between instances of" in stderr or "Unable to parse string" in stderr:
            return (
                "Verifier diagnosis: a column you treated as numeric is still text (object dtype). "
                "Coerce it with pd.to_numeric(..., errors='coerce') and handle the NaN before "
                "comparing or aggregating. Do not use .astype(..., errors='ignore')."
            )
        if "could not convert string to float" in stderr or "to numeric" in stderr:
            return (
                "Verifier diagnosis: you tried to convert a categorical text column to numbers. "
                "Only coerce columns that genuinely hold numeric values; encode categories with "
                "pd.get_dummies or groupby instead."
            )
        if "Input X contains NaN" in stderr:
            return (
                "Verifier diagnosis: feature columns still contain NaN. Add explicit missing-value "
                "handling before train_test_split/model.fit (for example, fill Age with its "
                "median). "
                "Keep the task-required LinearRegression model."
            )
        if "SyntaxError" in stderr and "@" in stderr:
            return "Verifier diagnosis: wrap the required answer tag in a Python print call."
        return ""

    @staticmethod
    def semantic_violation(task: BenchmarkTask, code: str) -> str:
        if "linear regression" in task.instruction.casefold() and "LinearRegression" not in code:
            return (
                "Policy violation: the task explicitly requires sklearn LinearRegression. "
                "Do not substitute LogisticRegression or another estimator."
            )
        return ""

    def _run_container(self, work_dir: Path, script_name: str) -> subprocess.CompletedProcess[str]:
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--memory",
            "768m",
            "--cpus",
            "1",
            "--pids-limit",
            "128",
            "--mount",
            f"type=bind,source={work_dir.resolve()},target=/workspace,readonly",
            self.config.image,
            f"/workspace/{script_name}",
        ]
        return subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )

    async def execute(
        self, task: BenchmarkTask, prepared: PreparedTask, handle: EnvironmentHandle
    ) -> TaskOutput:
        if prepared.task != task or prepared.work_dir != handle.work_dir:
            raise ValueError("data-analysis executor received mismatched task environment")
        started_at = monotonic()
        input_tokens = 0
        output_tokens = 0
        artifacts: list[ArtifactRef] = []
        feedback = ""
        final_stdout = ""
        attempts = 0
        invalid_tool_calls = 0
        inspect_path = prepared.work_dir / "inspect_table.py"
        inspect_path.write_text(
            "import pandas as pd\n"
            "df = pd.read_csv('/workspace/input.csv')\n"
            "print('COLUMNS:', list(df.columns))\n"
            "print('DTYPES:')\n"
            "print(df.dtypes.to_string())\n"
            "print('MISSING:', df.isna().sum().to_dict())\n"
            "print('HEAD:')\n"
            "print(df.head(3).to_csv(index=False))\n",
            encoding="utf-8",
        )
        inspection = await asyncio.to_thread(
            self._run_container, prepared.work_dir, inspect_path.name
        )
        if inspection.returncode != 0:
            raise RuntimeError(f"read_table inspection failed: {inspection.stderr.strip()}")
        table_profile = inspection.stdout.strip()
        artifacts.append(
            ArtifactRef(
                str(inspect_path),
                hashlib.sha256(inspect_path.read_bytes()).hexdigest(),
                "text/x-python",
            )
        )
        for attempt in range(1, self.config.max_attempts + 1):
            attempts = attempt
            request_text = (
                f"Task:\n{task.instruction}\n\n"
                "The CSV is available only as /workspace/input.csv. Write a complete Python "
                "program using pandas/scikit-learn as needed. Print only the required @name[value] "
                "answer tag(s). A bare @name[value] expression is invalid Python: emit every tag "
                "with print(f'@name[{value}]'). Do not use the network or write files. Return only "
                f"Python code.\n\nread_table output:\n{table_profile}"
            )
            if self.config.dtype_guardrail:
                request_text += f"\n\n{self.NUMERIC_GUARDRAIL}"
            if self.config.parse_robustness_guardrail:
                request_text += f"\n\n{self.PARSE_ROBUSTNESS_GUARDRAIL}"
            if feedback:
                request_text += f"\n\nPrevious execution feedback:\n{feedback}"
            response = await self.provider.complete(
                ModelRequest(
                    messages=(
                        Message("system", "You write deterministic data-analysis Python programs."),
                        Message("user", request_text),
                    ),
                    tools=(),
                    temperature=0.0,
                    max_output_tokens=self.config.max_output_tokens,
                )
            )
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            code = self.extract_python(response.content)
            script_path = prepared.work_dir / f"solution_attempt_{attempt}.py"
            script_path.write_text(code + "\n", encoding="utf-8")
            execution_path = prepared.work_dir / f"execution_attempt_{attempt}.json"
            violation = self.semantic_violation(task, code)
            if violation:
                stdout = ""
                stderr = violation
                return_code = 2
                invalid_tool_calls += 1
            else:
                try:
                    completed = await asyncio.to_thread(
                        self._run_container, prepared.work_dir, script_path.name
                    )
                    stdout = completed.stdout.strip()
                    stderr = completed.stderr.strip()
                    return_code = completed.returncode
                except subprocess.TimeoutExpired as error:
                    stdout = str(error.stdout or "").strip()
                    stderr = "container execution timed out"
                    return_code = 124
            execution_path.write_text(
                json.dumps(
                    {"return_code": return_code, "stdout": stdout, "stderr": stderr},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            for path, media_type in (
                (script_path, "text/x-python"),
                (execution_path, "application/json"),
            ):
                artifacts.append(
                    ArtifactRef(
                        str(path), hashlib.sha256(path.read_bytes()).hexdigest(), media_type
                    )
                )
            final_stdout = stdout
            if return_code == 0 and "@" in stdout:
                break
            feedback = (
                f"{self.diagnostic_hint(stderr)}\nReturn code: {return_code}\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr[-2000:]}\n"
                f"Previous code:\n{code}"
            )
        return TaskOutput(
            final_answer=final_stdout,
            structured_payload={"docker_image": self.config.image, "attempts": attempts},
            artifacts=tuple(artifacts),
            run_metrics=SharedRunMetrics(
                iterations=attempts,
                tool_calls=attempts + 1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                wall_seconds=monotonic() - started_at,
                recovery_count=max(0, attempts - 1),
                invalid_tool_calls=invalid_tool_calls,
            ),
        )
