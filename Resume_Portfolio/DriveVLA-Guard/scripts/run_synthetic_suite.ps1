$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$manifest = Join-Path $projectRoot "data\synthetic\smoke_manifest.jsonl"

python -m drivevla_guard.cli make-synthetic --output $manifest --scenes 12

$runs = @(
    @("b0", "configs\b0_fast_greedy.yaml"),
    @("b1", "configs\b1_slow_greedy.yaml"),
    @("e1", "configs\e1_candidates_only.yaml"),
    @("e2", "configs\e2_rerank.yaml"),
    @("e3", "configs\e3_router.yaml"),
    @("e4", "configs\e4_guard.yaml")
)

foreach ($run in $runs) {
    $name = $run[0]
    $config = Join-Path $projectRoot $run[1]
    $output = Join-Path $projectRoot "artifacts\runs\reproduction\$name\predictions.jsonl"
    $summary = Join-Path $projectRoot "artifacts\runs\reproduction\$name\summary.json"
    python -m drivevla_guard.cli evaluate-synthetic --config $config --manifest $manifest --output $output --summary $summary
}

python -m pytest (Join-Path $projectRoot "tests")

